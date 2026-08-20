# Clone Baseline NR Scenario into 5 Network Scenario Files

Create 5 scenario files in `contrib/nr/examples/`, each modeling a different network condition while sharing identical timing parameters and the `SampleFlowStats()` function for KPI tracing.

## Research Summary

### References Found

| Scenario | Reference File(s) | Key APIs |
|---|---|---|
| **0: Stable** | [cttc-nr-demo.cc](file:///home/thentt/ns-3-dev/contrib/nr/examples/cttc-nr-demo.cc) (baseline) | Unchanged |
| **1: HandoverStorm** | [nr-test-x2-handover.cc](file:///home/thentt/ns-3-dev/contrib/nr/test/nr-test-x2-handover.cc) | `NrA3RsrpHandoverAlgorithm`, `AddX2Interface`, `HandoverRequest`, `HandoverStart` trace |
| **2: NTNGap** | [gsoc-leo-demo-example.cc](file:///home/thentt/ns-3-dev/contrib/nr/examples/gsoc-leo-demo-example.cc) (**confirmed exists**) | `LeoOrbitNodeHelper`, `GeocentricConstantPositionMobilityModel`, NTN channel |
| **3: THzBlockage** | [nr-channel-helper.cc](file:///home/thentt/ns-3-dev/contrib/nr/helper/nr-channel-helper.cc), [channel-condition-model.h](file:///home/thentt/ns-3-dev/src/propagation/model/channel-condition-model.h) | `NrChannelHelper::ConfigureFactories("UMi", "LOS"/"NLOS")` — but this sets at init time. Use `NeverLosChannelConditionModel`/`AlwaysLosChannelConditionModel` for static. For dynamic toggle: increase pathloss via scheduled attribute changes. |
| **4: ISACContention** | Baseline + extra traffic | `UdpClientHelper`, `Simulator::Schedule` |

### Key Findings

1. **Handover (File 1)**: The NR module has `NrA3RsrpHandoverAlgorithm` with `Hysteresis` (dB) and `TimeToTrigger` attributes. The test [nr-test-x2-handover.cc](file:///home/thentt/ns-3-dev/contrib/nr/test/nr-test-x2-handover.cc) uses `NrNoOpHandoverAlgorithm` with manual `HandoverRequest()` calls and `ConstantPositionMobilityModel` + teleporting. For our "handover storm", I will use **manual scripted handovers** (same approach as the test) — this is more reliable than depending on A3 measurements with mobility, which could be timing-sensitive. The `HandoverStart` trace source is at `NrGnbRrc::m_handoverStartTrace` (Config path: `/NodeList/*/DeviceList/*/$ns3::NrGnbNetDevice/NrGnbRrc/HandoverStart`).

2. **NTN (File 2)**: `gsoc-leo-demo-example.cc` confirmed present. Uses `LeoOrbitNodeHelper` for orbital mobility, `GeocentricConstantPositionMobilityModel` for ground node, NTN channel scenario. `simTime` is 4s by default — will adjust to match baseline 1s. File does NOT have FlowMonitor — need to add.

3. **THzBlockage (File 3)**: No direct "force channel condition" API at runtime. The `ChannelConditionModel` is set once during channel creation. **Strategy**: Use `FixedRssLossModel` as an additional propagation loss model, or more practically, use `Simulator::Schedule` to periodically increase/decrease the `TxPower` of the gNB to simulate blockage windows (reducing effective SINR). Alternative: Create a custom approach using `Config::Set` to dynamically change the gNB's `TxPower` attribute during simulation.

4. **ISACContention (File 4)**: Straightforward — add a 3rd UdpClient with high packet rate, scheduled on/off with sinusoidal pattern.

> [!IMPORTANT]
> **File 3 (THzBlockage) design decision**: Since we cannot dynamically swap the ChannelConditionModel at runtime, I will simulate blockage by scheduling periodic `TxPower` reduction on the gNB (e.g., dropping from 35 dBm to 5 dBm during "blockage" windows). This effectively simulates severe signal degradation without requiring unsupported API calls. The blockage pattern follows the `step % 15 < 3` cadence from `env.py`.

## Proposed Changes

### CMakeLists.txt

#### [MODIFY] [CMakeLists.txt](file:///home/thentt/ns-3-dev/contrib/nr/examples/CMakeLists.txt)

Add the 5 new scenario names to the `base_examples` list:
- `cttc-nr-scenario-stable`
- `cttc-nr-scenario-handover-storm`
- `cttc-nr-scenario-ntn-gap`
- `cttc-nr-scenario-thz-blockage`
- `cttc-nr-scenario-isac-contention`

---

### File 0 — Stable Baseline

#### [NEW] [cttc-nr-scenario-stable.cc](file:///home/thentt/ns-3-dev/contrib/nr/examples/cttc-nr-scenario-stable.cc)

Exact copy of `cttc-nr-demo.cc` with:
- CSV filename changed to `kpi_trace_scenario0_stable.csv`
- Log component name changed to `CttcNrScenarioStable`

---

### File 1 — Handover Storm

#### [NEW] [cttc-nr-scenario-handover-storm.cc](file:///home/thentt/ns-3-dev/contrib/nr/examples/cttc-nr-scenario-handover-storm.cc)

Based on baseline, with these modifications:
- **3 gNBs** placed 500m apart using `ListPositionAllocator` (not `GridScenarioHelper`, since we need precise multi-gNB positioning with handover). Each gNB gets its own band (different frequencies to avoid inter-cell interference issues).
- **Manual handover scheduling**: Inspired by `nr-test-x2-handover.cc`, use `NrNoOpHandoverAlgorithm` and script ~6 handover events (back-and-forth between gNBs every ~100ms).
- **X2 interface**: `nrHelper->AddX2Interface(gnbNodes)`
- **UE teleportation**: Before each handover, teleport UE near target gNB (same pattern as test).
- **Handover counter**: Connect to `HandoverStart` trace source via `Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::NrGnbNetDevice/NrGnbRrc/HandoverStart", ...)` to count handover events.
- CSV: `kpi_trace_scenario1_handoverstorm.csv`

---

### File 2 — NTN Gap

#### [NEW] [cttc-nr-scenario-ntn-gap.cc](file:///home/thentt/ns-3-dev/contrib/nr/examples/cttc-nr-scenario-ntn-gap.cc)

Based on `gsoc-leo-demo-example.cc`, with modifications:
- Add `SampleFlowStats()` function and `FlowMonitorHelper` (file currently has no FlowMonitor)
- Change traffic from finite (10 packets) to continuous CBR (matching baseline's `lambdaULL`/`lambdaBe` patterns)
- Set `simTime = 1s`, `udpAppStartTime = 400ms`, `snapshotInterval = 0.1s` to match other scenarios
- CSV: `kpi_trace_scenario2_ntngap.csv`
- Keep all NTN-specific setup (orbital mobility, NTN channel, antenna orientation)

---

### File 3 — THz Blockage

#### [NEW] [cttc-nr-scenario-thz-blockage.cc](file:///home/thentt/ns-3-dev/contrib/nr/examples/cttc-nr-scenario-thz-blockage.cc)

Based on baseline, with additions:
- A `ToggleBlockage()` function scheduled periodically to simulate blockage windows
- Pattern: blockage active when `step % 15 < 3` (from `env.py`). With `simTime=1s` and 100ms snapshot intervals, that's ~10 steps total. Map: steps 0-2 blockage, steps 3-14 normal, etc. In 1s simulation: blockage at 0-0.2s (first 3 steps worth), normal rest.
- During blockage: set gNB TxPower to very low (5 dBm) via `NrHelper::GetGnbPhy(gnbNetDev.Get(0), 0)->SetTxPower(5.0)` and same for BWP 1
- During normal: restore TxPower to original value
- CSV: `kpi_trace_scenario3_thzblockage.csv`

---

### File 4 — ISAC Contention

#### [NEW] [cttc-nr-scenario-isac-contention.cc](file:///home/thentt/ns-3-dev/contrib/nr/examples/cttc-nr-scenario-isac-contention.cc)

Based on baseline, with additions:
- Add 1 extra UE + traffic flow (3rd `UdpClientHelper` with high packet rate `lambdaNoise = 20000`)
- `AdjustInterferenceTraffic()` function scheduled every 50ms to adjust the interference traffic `Interval` attribute using formula: `interval = 1.0 / (lambdaNoise * (0.3 + 0.3*sin(t*0.5)^2))`
- The 3rd flow is **not** logged in `SampleFlowStats` — achieved by filtering by flow ID (only log flows 1 and 2, skip flow 3+)
- CSV: `kpi_trace_scenario4_isaccontention.csv`

---

## Timing Parameters (identical across all 5 files)

| Parameter | Value |
|---|---|
| `simTime` | `MilliSeconds(1000)` |
| `udpAppStartTime` | `MilliSeconds(400)` |
| `snapshotInterval` | `0.1` (seconds) |

## Verification Plan

### Build
```bash
./ns3 build
```

### Run each scenario
```bash
./ns3 run cttc-nr-scenario-stable
./ns3 run cttc-nr-scenario-handover-storm
./ns3 run cttc-nr-scenario-ntn-gap
./ns3 run cttc-nr-scenario-thz-blockage
./ns3 run cttc-nr-scenario-isac-contention
```

### Check outputs
For each CSV file:
- Count total rows (expected: ~6 rows for 2 flows x 6 intervals from 0.5s to 1.0s, give or take)
- Print first 5 and last 5 rows
- Verify column schema: `time,flowId,throughput_mbps,delay_ms,jitter_ms,lost_packets`

### Scenario 1 specific
- Report handover count from trace callback
- Confirm handovers actually occurred (count > 0)
