// cmd/acp2p-emitter/main.go
package main

/*
Usage examples:

# Full sample with current arguments:
# - use either -ip or -ifname
# - use -app-id 0 for weighted random app selection, or 1..3 to force an app
# - use -manual-json only when you want startup manual mode
# - queued requests wait FIFO until the active lease completes
./buyer_v4 \
  -rpc-addr 127.0.0.1:7373 \
  -event buyer.request \
  -ip 10.0.1.11 \
  -ifname eth0 \
  -http-host 0.0.0.0 \
  -http-port 8090 \
  -lease-duration 120 \
  -arrival-lambda-per-hour 0.5 \
  -app-id 0 \
  -app1-pct 34 -app2-pct 33 -app3-pct 33 \
  -score-min 0.0 -score-max 1.0 \
  -budget-min 0.0 -budget-max 3.0 \
  -manual-json /opt/serfapp/buyer.json \
  -v

# Typical random app-based Poisson arrivals with FIFO lease queueing:
./buyer_v4 \
  -rpc-addr 127.0.0.1:7373 \
  -event buyer.request \
  -ifname eth0 \
  -http-host 0.0.0.0 -http-port 8090 \
  -arrival-lambda-per-hour 0.5 \
  -app-id 0 \
  -app1-pct 34 -app2-pct 33 -app3-pct 33 \
  -score-min 0.0 -score-max 1.0 \
  -budget-min 0.0 -budget-max 3.0

# Behavior:
# - a new request arrival is sampled from a Poisson process
# - App 1 / App 2 / App 3 is selected by the configured weights
# - if a lease is still active, the new request is queued FIFO
# - once the active lease completes, the next queued request starts

# Manual app selection:
./buyer_v4 \
  -rpc-addr 127.0.0.1:7373 \
  -event buyer.request \
  -ifname eth0 \
  -http-host 0.0.0.0 -http-port 8090 \
  -app-id 2

# Manual at startup (JSON file):
./buyer_v4 \
  -rpc-addr 127.0.0.1:7373 \
  -event buyer.request \
  -ifname eth0 \
  -http-host 0.0.0.0 -http-port 8090 \
  -lease-duration 120 \
  -manual-json /opt/serfapp/buyer.json

# Manual at runtime (HTTP):
POST http://<host>:8090/buyer
Content-Type: application/json
{
  "ip": "10.0.1.11",
  "lease_duration": 120,
  "resources": {
    "vcpu":    {"demand_per_unit": 2, "score": 2.2, "budget": 2.5},
    "ram":     {"demand_per_unit": 4, "score": 1.8, "budget": 2.3},
    "storage": {"demand_per_unit": 8, "score": 2.0, "budget": 2.2},
    "vgpu":    {"demand_per_unit": 0, "score": 0.0, "budget": 0.0}
  }
}

# Clear manual mode (resume random):
POST http://<host>:8090/buyer/manual/clear
*/

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	serfclient "github.com/hashicorp/serf/client"
)

// ---------------- Shared schema ----------------
type BuyerResource struct {
	DemandPerUnit int     `json:"demand_per_unit"`
	Score         float64 `json:"score"`
	Budget        float64 `json:"budget"`
}
type BuyerRequest struct {
	IP            string                   `json:"ip"`
	LeaseDuration int                      `json:"lease_duration,omitempty"`
	Resources     map[string]BuyerResource `json:"resources"`
	Time          string                   `json:"time,omitempty"`
}

type AppProfile struct {
	ID            int
	Name          string
	LeaseDuration int
	CPU           int
	RAM           int
	Storage       int
	GPU           int
}

var appProfiles = map[int]AppProfile{
	1: {ID: 1, Name: "Social Network", LeaseDuration: 14400, CPU: 40, RAM: 54, Storage: 14, GPU: 0},
	2: {ID: 2, Name: "Sentiment Analysis", LeaseDuration: 600, CPU: 13, RAM: 14, Storage: 17, GPU: 0},
	3: {ID: 3, Name: "Hotel Reservation", LeaseDuration: 18000, CPU: 20, RAM: 2, Storage: 12, GPU: 0},
}

// ---------------- Logging helpers ----------------
func infof(format string, a ...any) { log.Printf("[INFO] "+format, a...) }
func warnf(format string, a ...any) { log.Printf("[WARN] "+format, a...) }
func errf(format string, a ...any)  { log.Printf("[ERROR] "+format, a...) }
func dbg(format string, a ...any)   { log.Printf("[DEBUG] "+format, a...) }

// ---------------- Small utils ----------------
func ipv4ForInterface(ifName string) (string, error) {
	ifi, err := net.InterfaceByName(ifName)
	if err != nil {
		return "", fmt.Errorf("interface %q: %w", ifName, err)
	}
	addrs, err := ifi.Addrs()
	if err != nil {
		return "", fmt.Errorf("interface %q addrs: %w", ifName, err)
	}
	for _, a := range addrs {
		var ip net.IP
		switch v := a.(type) {
		case *net.IPNet:
			ip = v.IP
		case *net.IPAddr:
			ip = v.IP
		}
		if ip == nil || ip.IsLoopback() {
			continue
		}
		if p4 := ip.To4(); p4 != nil {
			return p4.String(), nil
		}
	}
	return "", fmt.Errorf("no IPv4 found on interface %q", ifName)
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }
func uniform2(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func poissonArrivalDelay(lambdaPerHour float64) time.Duration {
	if lambdaPerHour <= 0 {
		return time.Hour
	}
	hours := rand.ExpFloat64() / lambdaPerHour
	return time.Duration(hours * float64(time.Hour))
}

func pickAppProfile(manualID int, p1, p2, p3 float64) (AppProfile, error) {
	if manualID != 0 {
		prof, ok := appProfiles[manualID]
		if !ok {
			return AppProfile{}, fmt.Errorf("invalid app-id %d (expected 1, 2, or 3)", manualID)
		}
		return prof, nil
	}

	total := p1 + p2 + p3
	if total <= 0 {
		return AppProfile{}, fmt.Errorf("app percentages must sum to a positive value")
	}
	draw := rand.Float64() * total
	if draw < p1 {
		return appProfiles[1], nil
	}
	draw -= p1
	if draw < p2 {
		return appProfiles[2], nil
	}
	return appProfiles[3], nil
}

// ---------------- HTTP state ----------------
type latestStore struct {
	mu     sync.RWMutex
	curr   BuyerRequest
	manual int32 // 0=false, 1=true
}

func (s *latestStore) set(br BuyerRequest) {
	s.mu.Lock()
	s.curr = br
	s.mu.Unlock()
}
func (s *latestStore) get() BuyerRequest {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.curr
}
func (s *latestStore) setManual(on bool) {
	if on {
		atomic.StoreInt32(&s.manual, 1)
	} else {
		atomic.StoreInt32(&s.manual, 0)
	}
}
func (s *latestStore) isManual() bool { return atomic.LoadInt32(&s.manual) == 1 }

func startLeaseTimer(timer **time.Timer, leaseC *<-chan time.Time, leaseDurationSeconds int) {
	if *timer != nil {
		if !(*timer).Stop() {
			select {
			case <-*leaseC:
			default:
			}
		}
	}

	if leaseDurationSeconds <= 0 {
		leaseDurationSeconds = 1
	}
	*timer = time.NewTimer(time.Duration(leaseDurationSeconds) * time.Second)
	*leaseC = (*timer).C
}

func resetArrivalTimer(timer *time.Timer, nextDelay time.Duration) {
	if nextDelay < 0 {
		nextDelay = 0
	}
	if !timer.Stop() {
		select {
		case <-timer.C:
		default:
		}
	}
	timer.Reset(nextDelay)
}

func summarizeQueueRequest(br BuyerRequest) string {
	vcpu := br.Resources["vcpu"].DemandPerUnit
	ram := br.Resources["ram"].DemandPerUnit
	storage := br.Resources["storage"].DemandPerUnit
	vgpu := br.Resources["vgpu"].DemandPerUnit
	return fmt.Sprintf("ip=%s lease=%ds demand=(vcpu=%d ram=%d storage=%d vgpu=%d)",
		br.IP, br.LeaseDuration, vcpu, ram, storage, vgpu)
}

func logQueueState(active *BuyerRequest, queue []BuyerRequest) {
	if active == nil {
		infof("queue state: active=<none> queued=%d", len(queue))
	} else {
		infof("queue state: active={%s} queued=%d", summarizeQueueRequest(*active), len(queue))
	}
	for i, br := range queue {
		infof("queue[%d]: {%s}", i, summarizeQueueRequest(br))
	}
}

// ---------------- Main ----------------
func main() {
	rand.Seed(time.Now().UnixNano())
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	var (
		// Serf
		rpcAddr   = flag.String("rpc-addr", "127.0.0.1:7373", "Serf RPC address")
		eventName = flag.String("event", "buyer.request", "Serf user-event name")

		// Identity
		ip     = flag.String("ip", "", "Buyer IP (required if -ifname empty)")
		ifname = flag.String("ifname", "", "Interface to auto-detect IPv4 (used if -ip empty)")

		// HTTP
		httpHost = flag.String("http-host", "0.0.0.0", "HTTP bind host")
		httpPort = flag.Int("http-port", 8090, "HTTP bind port")
		leaseDur = flag.Int("lease-duration", 120, "Lease duration to include in emitted buyer requests")

		// Arrival / application selection
		arrivalLambdaPerHour = flag.Float64("arrival-lambda-per-hour", 0.5, "Poisson arrival rate in requests per hour")
		appID                = flag.Int("app-id", 0, "Force a specific application profile (0=random, 1..3=manual)")
		app1Pct              = flag.Float64("app1-pct", 34.0, "Selection weight/percentage for App 1 (Social Network)")
		app2Pct              = flag.Float64("app2-pct", 33.0, "Selection weight/percentage for App 2 (Sentiment Analysis)")
		app3Pct              = flag.Float64("app3-pct", 33.0, "Selection weight/percentage for App 3 (Hotel Reservation)")

		// Score & budget bounds (uniform in [min,max], rounded to 2 dp)
		scoreMin = flag.Float64("score-min", 0.0, "Min score")
		scoreMax = flag.Float64("score-max", 1.0, "Max score")
		budMin   = flag.Float64("budget-min", 0.0, "Min budget")
		budMax   = flag.Float64("budget-max", 3.0, "Max budget")

		// Manual input at startup (optional)
		manualJSON = flag.String("manual-json", "", "If provided, load BuyerRequest JSON and switch to manual mode")
		verbose    = flag.Bool("v", false, "Verbose debug logging")
	)
	flag.Parse()

	if *ip == "" && *ifname == "" {
		log.Fatal("must provide -ip or -ifname")
	}
	if *ip == "" {
		autoIP, err := ipv4ForInterface(*ifname)
		if err != nil {
			log.Fatalf("auto-detect IP from -ifname=%s: %v", *ifname, err)
		}
		*ip = autoIP
	}
	if *verbose {
		infof("starting app-based emitter ip=%s serf=%s event=%s http=%s:%d arrival_lambda_per_hour=%.3f app_id=%d app_pcts=[%.2f,%.2f,%.2f]",
			*ip, *rpcAddr, *eventName, *httpHost, *httpPort, *arrivalLambdaPerHour, *appID, *app1Pct, *app2Pct, *app3Pct)
	}

	// Serf RPC
	rc, err := serfclient.NewRPCClient(*rpcAddr)
	if err != nil {
		log.Fatalf("serf RPC connect: %v", err)
	}
	defer rc.Close()

	// Latest request store
	store := &latestStore{}

	// If manual JSON provided at startup, load it and enable manual mode.
	if *manualJSON != "" {
		f, err := os.Open(*manualJSON)
		if err != nil {
			log.Fatalf("open -manual-json: %v", err)
		}
		defer f.Close()
		var br BuyerRequest
		if err := json.NewDecoder(f).Decode(&br); err != nil {
			log.Fatalf("decode -manual-json: %v", err)
		}
		normalizeManual(&br, *ip, *leaseDur)
		store.set(br)
		store.setManual(true)
		infof("manual mode enabled from %s", *manualJSON)
	}

	// HTTP server
	mux := http.NewServeMux()

	// GET returns current request (manual or last random).
	mux.HandleFunc("/buyer", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			br := store.get()
			if br.IP == "" {
				http.Error(w, "no demand generated yet", http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(br)
		case http.MethodPost:
			// POST sets manual request and turns on manual mode.
			body, err := io.ReadAll(r.Body)
			if err != nil {
				http.Error(w, "read body error", http.StatusBadRequest)
				return
			}
			var br BuyerRequest
			if err := json.Unmarshal(body, &br); err != nil {
				http.Error(w, "invalid JSON", http.StatusBadRequest)
				return
			}
			normalizeManual(&br, *ip, *leaseDur)
			if len(br.Resources) == 0 {
				http.Error(w, "resources required", http.StatusBadRequest)
				return
			}
			store.set(br)
			store.setManual(true)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"status":      "ok",
				"manual_mode": true,
				"buyer":       br,
			})
			infof("manual mode enabled via POST /buyer")
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	// POST to clear manual mode (resume random generation).
	mux.HandleFunc("/buyer/manual/clear", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		store.setManual(false)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"status": "ok", "manual_mode": false})
		infof("manual mode cleared (random generation will resume)")
	})

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	srv := &http.Server{
		Addr:              fmt.Sprintf("%s:%d", *httpHost, *httpPort),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		infof("HTTP serving demand at http://%s:%d/buyer", *httpHost, *httpPort)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errf("http server error: %v", err)
		}
	}()

	// Graceful shutdown
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	defer func() {
		shCtx, shCancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer shCancel()
		_ = srv.Shutdown(shCtx)
	}()

	startRequest := func(br BuyerRequest) BuyerRequest {
		started := br
		started.Time = time.Now().Format(time.RFC3339)

		// Broadcast Serf user event
		payload, err := json.Marshal(started)
		if err != nil {
			warnf("marshal buyer request for user-event failed: %v", err)
		} else if err := rc.UserEvent(*eventName, payload, false); err != nil {
			warnf("send user-event %q failed: %v", *eventName, err)
		} else {
			infof("broadcast user:%s ip=%s at %s lease_duration=%ds (manual=%v)",
				*eventName, started.IP, started.Time, started.LeaseDuration, store.isManual())
		}

		// JUST FOR NOW:: QUICK FIX. NEEDS TO BE CHANGED LATER
		triggerPayload, err := json.Marshal(started)
		if err != nil {
			warnf("marshal buyer request for trigger failed: %v", err)
			return started
		}

		go func(triggerPayload []byte, triggerTime string, manual bool) {
			resp, err := http.Post(
				"http://localhost:4041/trigger",
				"application/json",
				bytes.NewBuffer(triggerPayload),
			)
			if err != nil {
				warnf("trigger POST failed at %s (manual=%v): %v", triggerTime, manual, err)
				return
			}
			defer resp.Body.Close()

			body, readErr := io.ReadAll(resp.Body)
			if readErr != nil {
				warnf("trigger response read failed at %s status=%d (manual=%v): %v",
					triggerTime, resp.StatusCode, manual, readErr)
				return
			}

			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				infof("trigger POST succeeded at %s status=%d (manual=%v) payload=%s body=%s",
					triggerTime, resp.StatusCode, manual, string(triggerPayload), string(body))
				return
			}

			warnf("trigger POST unsuccessful at %s status=%d (manual=%v) payload=%s body=%s",
				triggerTime, resp.StatusCode, manual, string(triggerPayload), string(body))
		}(triggerPayload, started.Time, store.isManual())
		return started
	}

	nextRequest := func() (BuyerRequest, string) {
		if store.isManual() {
			br := store.get()
			normalizeManual(&br, *ip, *leaseDur)
			store.set(br)
			return br, "manual"
		}

		prof, err := pickAppProfile(*appID, *app1Pct, *app2Pct, *app3Pct)
		if err != nil {
			log.Fatalf("select app profile: %v", err)
		}
		br := makeAppBuyer(*ip, prof, *scoreMin, *scoreMax, *budMin, *budMax)
		store.set(br)
		infof("generated app=%d (%s) demand=(vcpu=%d ram=%d storage=%d vgpu=%d) lease_duration=%ds",
			prof.ID, prof.Name, prof.CPU, prof.RAM, prof.Storage, prof.GPU, prof.LeaseDuration)
		return br, fmt.Sprintf("app=%d (%s)", prof.ID, prof.Name)
	}

	var (
		arrivalTimer = time.NewTimer(0)
		leaseTimer   *time.Timer
		leaseC       <-chan time.Time
		active       *BuyerRequest
		activeUntil  time.Time
		queue        []BuyerRequest
	)
	defer arrivalTimer.Stop()
	defer func() {
		if leaseTimer != nil {
			leaseTimer.Stop()
		}
	}()

	// Event loop: arrivals generate requests; lease completion advances the FIFO queue.
	for {
		select {
		case <-ctx.Done():
			infof("shutting down (signal received)")
			return
		case <-arrivalTimer.C:
			br, source := nextRequest()
			if active == nil {
				started := startRequest(br)
				activeBR := started
				active = &activeBR
				activeUntil = time.Now().Add(time.Duration(started.LeaseDuration) * time.Second)
				startLeaseTimer(&leaseTimer, &leaseC, started.LeaseDuration)
				infof("started demand immediately from %s; lease active for %ds", source, started.LeaseDuration)
				logQueueState(active, queue)
			} else {
				queue = append(queue, br)
				remaining := time.Until(activeUntil)
				if remaining < 0 {
					remaining = 0
				}
				infof("queued demand from %s at position=%d while active lease has ~%s remaining",
					source, len(queue), remaining.Truncate(time.Second))
				logQueueState(active, queue)
			}

			wait := poissonArrivalDelay(*arrivalLambdaPerHour)
			infof("next request arrival in %s (lambda=%.3f req/hour, queue_len=%d)",
				wait, *arrivalLambdaPerHour, len(queue))
			resetArrivalTimer(arrivalTimer, wait)
		case <-leaseC:
			if active != nil {
				infof("lease completed for ip=%s demand_time=%s queue_len=%d", active.IP, active.Time, len(queue))
			}
			if len(queue) == 0 {
				active = nil
				leaseC = nil
				leaseTimer = nil
				infof("no queued demand waiting; scheduler is idle until the next arrival")
				logQueueState(active, queue)
				continue
			}

			next := queue[0]
			queue = queue[1:]
			started := startRequest(next)
			activeBR := started
			active = &activeBR
			activeUntil = time.Now().Add(time.Duration(started.LeaseDuration) * time.Second)
			startLeaseTimer(&leaseTimer, &leaseC, started.LeaseDuration)
			infof("started queued demand; remaining_queue_len=%d lease_duration=%ds", len(queue), started.LeaseDuration)
			logQueueState(active, queue)
		}
	}
}

// normalizeManual ensures IP and timestamp are set; keeps user-provided resources intact.
func normalizeManual(br *BuyerRequest, defaultIP string, defaultLeaseDuration int) {
	if br.IP == "" {
		br.IP = defaultIP
	}
	if br.LeaseDuration == 0 {
		br.LeaseDuration = defaultLeaseDuration
	}
	if br.Resources == nil {
		br.Resources = map[string]BuyerResource{}
	}
	// Ensure canonical keys exist (optional; leave as-is if user omits).
	need := []string{"vcpu", "ram", "storage", "vgpu"}
	for _, k := range need {
		if _, ok := br.Resources[k]; !ok {
			// keep missing keys absent; discovery side should tolerate zeros/missing.
			continue
		}
	}
	br.Time = time.Now().Format(time.RFC3339)
}

func makeAppBuyer(ip string, prof AppProfile, sMin, sMax, bMin, bMax float64) BuyerRequest {
	now := time.Now().Format(time.RFC3339)

	score := func() float64 { return round2(uniform2(sMin, sMax)) }
	budget := func() float64 { return round2(uniform2(bMin, bMax)) }

	return BuyerRequest{
		IP:            ip,
		LeaseDuration: prof.LeaseDuration,
		Time:          now,
		Resources: map[string]BuyerResource{
			"storage": {DemandPerUnit: prof.Storage, Score: score(), Budget: budget()},
			"vcpu":    {DemandPerUnit: prof.CPU, Score: score(), Budget: budget()},
			"ram":     {DemandPerUnit: prof.RAM, Score: score(), Budget: budget()},
			"vgpu":    {DemandPerUnit: prof.GPU, Score: score(), Budget: budget()},
		},
	}
}
