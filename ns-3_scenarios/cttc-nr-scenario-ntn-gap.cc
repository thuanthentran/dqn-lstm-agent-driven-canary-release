// Copyright (c) 2025 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

/**
 * @ingroup examples
 * @file cttc-nr-scenario-ntn-gap.cc
 * @brief Scenario 2 (NTNGap): LEO satellite NTN link with KPI tracing.
 *
 * Based on gsoc-leo-demo-example.cc. Adds FlowMonitor and SampleFlowStats()
 * for periodic KPI CSV output. Adjusts timing to match other scenarios.
 */

#include "math.h"

#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/geocentric-constant-position-mobility-model.h"
#include "ns3/geographic-positions.h"
#include "ns3/internet-module.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/leo-orbit-node-helper.h"
#include "ns3/nr-helper.h"
#include "ns3/nr-module.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/packet-sink.h"
#include "ns3/udp-client-server-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("CttcNrScenarioNtnGap");

Vector groundNodePosGEO;
Vector groundNodePosECEF;

// Global PHY-layer telemetry
static double g_lastSinrDb = 20.0;
static double g_lastRsrpDbm = -75.0;

void RecordDlDataSinr(uint16_t cellId, uint16_t rnti, double sinr, uint16_t ccId) {
    g_lastSinrDb = (sinr > 0) ? 10.0 * std::log10(sinr) : -30.0;
}

void RecordReportRsrp(uint16_t p1, uint16_t p2, uint16_t p3, double rsrp, uint8_t ccId) {
    g_lastRsrpDbm = (rsrp > 0) ? 10.0 * std::log10(rsrp * 1000.0) : -140.0;
}


void
UpdateAntennaOrientation(Ptr<Node> node, Ptr<UniformPlanarArray> satelliteNodeAntenna, Time period)
{
    auto mobility = node->GetObject<MobilityModel>();
    const auto satelliteNodePositionECEF = mobility->GetPosition();

    const Vector satelliteNodePositionGEO =
        GeographicPositions::CartesianToGeographicCoordinates(satelliteNodePositionECEF,
                                                              GeographicPositions::SPHERE);

    const Vector translatedENU =
        GeographicPositions::GeographicToTopocentricCoordinates(groundNodePosGEO,
                                                                satelliteNodePositionGEO,
                                                                GeographicPositions::SPHERE);

    const Angles angles(translatedENU);

    satelliteNodeAntenna->SetAlpha(angles.GetAzimuth());
    satelliteNodeAntenna->SetBeta(angles.GetInclination());

    Simulator::Schedule(period, &UpdateAntennaOrientation, node, satelliteNodeAntenna, period);
}

void
SampleFlowStats(Ptr<FlowMonitor> monitor,
                 Ptr<Ipv4FlowClassifier> classifier,
                 std::map<FlowId, FlowMonitor::FlowStats>& lastStats,
                 std::ofstream& traceFile,
                 double interval)
{
    monitor->CheckForLostPackets();
    auto stats = monitor->GetFlowStats();
    double now = Simulator::Now().GetSeconds();

    for (auto& kv : stats)
    {
        FlowId id = kv.first;
        auto& cur = kv.second;
        auto it = lastStats.find(id);

        uint64_t dRxBytes = cur.rxBytes - (it != lastStats.end() ? it->second.rxBytes : 0);
        uint32_t dRxPackets = cur.rxPackets - (it != lastStats.end() ? it->second.rxPackets : 0);
        double dDelaySum = (cur.delaySum - (it != lastStats.end() ? it->second.delaySum : Time(0))).GetSeconds();
        double dJitterSum = (cur.jitterSum - (it != lastStats.end() ? it->second.jitterSum : Time(0))).GetSeconds();
        uint64_t dLost = cur.lostPackets - (it != lastStats.end() ? it->second.lostPackets : 0);

        double throughputMbps = (dRxBytes * 8.0) / interval / 1e6;
        double meanDelayMs = dRxPackets > 0 ? (dDelaySum / dRxPackets) * 1000.0 : 0.0;
        double meanJitterMs = dRxPackets > 0 ? (dJitterSum / dRxPackets) * 1000.0 : 0.0;
        double pktLossRate = (dRxPackets + dLost) > 0 ? (double)dLost / (dRxPackets + dLost) : 0.0;

        traceFile << now << "," << id << "," << throughputMbps << ","
                   << meanDelayMs << "," << meanJitterMs << "," << dLost
                   << "," << g_lastSinrDb << "," << g_lastRsrpDbm
                   << "," << pktLossRate << "\n";
    }
    lastStats = stats;

    Simulator::Schedule(Seconds(interval), &SampleFlowStats, monitor, classifier,
                         std::ref(lastStats), std::ref(traceFile), interval);
}

/**
 * @brief Apply per-application defaults for a representative NTN deployment.
 *
 * @param application use case: "dtm", "vsat" or "backhaul"
 * @param frequencyHz [out] carrier frequency in Hz
 * @param bandwidthHz [out] bandwidth in Hz
 * @param satEIRP [out] satellite EIRP density in dBW/MHz
 * @param groundTxPower [out] terminal transmit power in dBm
 * @param satAntennaGainDb [out] satellite antenna gain in dBi
 * @param vsatAntennaGainDb [out] terminal antenna gain in dBi
 * @param satNoiseFigureDb [out] satellite (gNB) receiver noise figure in dB
 * @param altitudeKm [out] constellation altitude in km
 */
void
ApplyApplicationPreset(const std::string& application,
                       double& frequencyHz,
                       double& bandwidthHz,
                       double& satEIRP,
                       double& groundTxPower,
                       double& satAntennaGainDb,
                       double& vsatAntennaGainDb,
                       double& satNoiseFigureDb,
                       double& altitudeKm)
{
    if (application == "dtm")
    {
        frequencyHz = 0.7e9;
        bandwidthHz = 5e6;
        satEIRP = 50;
        groundTxPower = 23;
        satAntennaGainDb = 60;
        vsatAntennaGainDb = 0;
        satNoiseFigureDb = 1.5;
        altitudeKm = 550;
    }
    else if (application == "vsat")
    {
        frequencyHz = 20e9;
        bandwidthHz = 100e6;
        satEIRP = 24;
        groundTxPower = 33;
        satAntennaGainDb = 38.5;
        vsatAntennaGainDb = 40;
        satNoiseFigureDb = 5.0;
        altitudeKm = 1200;
    }
    else if (application == "backhaul")
    {
        frequencyHz = 20e9;
        bandwidthHz = 400e6;
        satEIRP = 20;
        groundTxPower = 40;
        satAntennaGainDb = 38.5;
        vsatAntennaGainDb = 50;
        satNoiseFigureDb = 5.0;
        altitudeKm = 1200;
    }
    else
    {
        NS_ABORT_MSG("Unknown application '" << application
                                             << "'. Valid values: dtm, vsat, backhaul.");
    }
}

int
main(int argc, char* argv[])
{
    CommandLine cmd;
    std::string orbitFile;
    uint32_t precision = 1000;
    std::string scenario = "NTN-Rural";

    // Timing parameters — identical across all 5 scenarios
    // Timing parameters
    Time simTime = MilliSeconds(30000);
    Time udpAppStartTime = MilliSeconds(400);

    std::string application = "vsat";
    const std::string appOpt = "--application=";
    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        if (arg.rfind(appOpt, 0) == 0)
        {
            application = arg.substr(appOpt.size());
        }
    }
    double frequencyHz;
    double bandwidthHz;
    double satEIRP;
    double groundTxPower;
    double satAntennaGainDb;
    double vsatAntennaGainDb;
    double satNoiseFigureDb;
    double altitudeKm;
    ApplyApplicationPreset(application,
                           frequencyHz,
                           bandwidthHz,
                           satEIRP,
                           groundTxPower,
                           satAntennaGainDb,
                           vsatAntennaGainDb,
                           satNoiseFigureDb,
                           altitudeKm);
    bool realisticPower = false;

    cmd.AddValue("application",
                 "NTN use case preset: 'dtm', 'vsat' or 'backhaul'.",
                 application);
    cmd.AddValue("orbitFile", "CSV file with orbit parameters", orbitFile);
    cmd.AddValue("precision", "Mobility model time precision in milliseconds", precision);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.AddValue("scenario", "Scenario for the 3GPP Channel Model", scenario);
    cmd.AddValue("frequencyHz", "The operating frequency in Hz", frequencyHz);
    cmd.AddValue("bandwidthHz", "The bandwidth in Hz", bandwidthHz);
    cmd.AddValue("satAntennaGainDb", "Satellite Antenna Gain in dBi", satAntennaGainDb);
    cmd.AddValue("vsatAntennaGainDb", "VSAT Antenna Gain in dBi", vsatAntennaGainDb);
    cmd.AddValue("groundTxPower", "Ground Node TxPower in dBm", groundTxPower);
    cmd.AddValue("satEIRP", "Satellite EIRP in dBW/MHz", satEIRP);
    cmd.AddValue("satNoiseFigure", "Satellite noise figure in dB", satNoiseFigureDb);
    cmd.AddValue("altitudeKm", "Constellation altitude in km", altitudeKm);
    cmd.AddValue("realisticPower", "Compensate for antenna array gain", realisticPower);
    cmd.Parse(argc, argv);

    LeoOrbitNodeHelper orbit(Time(MilliSeconds(precision)));

    NodeContainer satellites;
    if (!orbitFile.empty())
    {
        satellites = orbit.CreateNodesAndInstallMobility(orbitFile);
    }
    else
    {
        satellites = orbit.CreateNodesAndInstallMobility(LeoOrbitalShell(altitudeKm, 30, 1, 2));
    }

    Ptr<Node> groundNode = CreateObject<Node>();
    Ptr<GeocentricConstantPositionMobilityModel> groundNodeMobility =
        CreateObject<GeocentricConstantPositionMobilityModel>();

    auto firstSatelliteMobility = satellites.Get(0)->GetObject<MobilityModel>();
    auto firstSatellitePositionGEO =
        GeographicPositions::CartesianToGeographicCoordinates(firstSatelliteMobility->GetPosition(),
                                                              GeographicPositions::SPHERE);
    groundNodeMobility->SetGeographicPosition(
        Vector(firstSatellitePositionGEO.x, firstSatellitePositionGEO.y, 0));

    groundNodePosGEO = groundNodeMobility->GetGeographicPosition();
    groundNodePosECEF = groundNodeMobility->GetGeocentricPosition();

    groundNode->AggregateObject(groundNodeMobility);

    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);

    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    constexpr uint8_t numCcPerBand = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf(frequencyHz, bandwidthHz, numCcPerBand);
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories(scenario, "Default", "ThreeGpp");
    channelHelper->AssignChannelsToBands({band});
    allBwps = CcBwpCreator::GetAllBwps({band});

    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));

    nrHelper->SetSchedulerTypeId(NrMacSchedulerTdmaRR::GetTypeId());

    const uint32_t ueNumRows = 2;
    const uint32_t ueNumCols = 4;
    const uint32_t gnbNumRows = 8;
    const uint32_t gnbNumCols = 8;
    const double ueArrayFactorDb = 10 * std::log10(ueNumRows * ueNumCols);
    const double gnbArrayFactorDb = 10 * std::log10(gnbNumRows * gnbNumCols);

    double ueElementGainDb = vsatAntennaGainDb;
    double gnbElementGainDb = satAntennaGainDb;
    if (realisticPower)
    {
        ueElementGainDb = vsatAntennaGainDb - ueArrayFactorDb;
        gnbElementGainDb = satAntennaGainDb - gnbArrayFactorDb;
    }

    nrHelper->SetUeAntennaTypeId("ns3::UniformPlanarArray");
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(ueNumRows));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(ueNumCols));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObjectWithAttributes<IsotropicAntennaModel>(
                                        "Gain",
                                        DoubleValue(ueElementGainDb))));

    nrHelper->SetGnbAntennaTypeId("ns3::UniformPlanarArray");
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(gnbNumRows));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(gnbNumCols));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObjectWithAttributes<IsotropicAntennaModel>(
                                         "Gain",
                                         DoubleValue(gnbElementGainDb))));

    NodeContainer groundNodeContainer;
    groundNodeContainer.Add(groundNode);
    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(satellites, allBwps);
    NetDeviceContainer groundNodeNetDev = nrHelper->InstallUeDevice(groundNodeContainer, allBwps);

    // --- PHY traces: SINR and RSRP (ns-3.48 / cttc-nr correct signatures) ---
    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*/$ns3::NrUeNetDevice/ComponentCarrierMapUe/*/NrUePhy/DlDataSinr",
        MakeCallback(&RecordDlDataSinr));

    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*/$ns3::NrUeNetDevice/ComponentCarrierMapUe/*/NrUePhy/ReportRsrp",
        MakeCallback(&RecordReportRsrp));

    int64_t randomStream = 1;
    randomStream += nrHelper->AssignStreams(gnbNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(groundNodeNetDev, randomStream);

    double satTxPower = (satEIRP + 30) + (10 * std::log10(bandwidthHz / 1e6));
    if (realisticPower)
    {
        satTxPower -= satAntennaGainDb;
    }

    for (uint32_t i = 0; i < gnbNetDev.GetN(); i++)
    {
        NrHelper::GetGnbPhy(gnbNetDev.Get(i), 0)->SetTxPower(satTxPower);
        NrHelper::GetGnbPhy(gnbNetDev.Get(i), 0)->SetNoiseFigure(satNoiseFigureDb);
    }

    NrHelper::GetUePhy(groundNodeNetDev.Get(0), 0)->SetTxPower(groundTxPower);

    auto [remoteHost, remoteHostIpv4Address] =
        nrEpcHelper->SetupRemoteHost("100Gb/s", 2500, MilliSeconds(10));

    InternetStackHelper internet;
    internet.Install(groundNodeContainer);

    Ipv4InterfaceContainer ueIpInterface;
    ueIpInterface = nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(groundNodeNetDev));

    // Configure continuous CBR traffic (matching baseline pattern)
    uint16_t dlPort = 1234;
    uint16_t ulPort = 1235;
    ApplicationContainer clientApps;
    ApplicationContainer serverApps;

    // DL server on ground node
    UdpServerHelper dlPacketSinkHelper(dlPort);
    serverApps.Add(dlPacketSinkHelper.Install(groundNode));

    // DL client: continuous CBR from remote host to ground node
    UdpClientHelper dlClientHelper;
    dlClientHelper.SetAttribute("Remote",
        AddressValue(addressUtils::ConvertToSocketAddress(ueIpInterface.GetAddress(0), dlPort)));
    dlClientHelper.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClientHelper.SetAttribute("PacketSize", UintegerValue(1252));
    dlClientHelper.SetAttribute("Interval", TimeValue(Seconds(1.0 / 10000)));
    clientApps.Add(dlClientHelper.Install(remoteHost));

    // UL server on remote host
    UdpServerHelper ulPacketSinkHelper(ulPort);
    serverApps.Add(ulPacketSinkHelper.Install(remoteHost));

    // UL client: continuous CBR from ground node to remote host
    UdpClientHelper ulClientHelper;
    ulClientHelper.SetAttribute("Remote",
        AddressValue(addressUtils::ConvertToSocketAddress(remoteHostIpv4Address, ulPort)));
    ulClientHelper.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    ulClientHelper.SetAttribute("PacketSize", UintegerValue(100));
    ulClientHelper.SetAttribute("Interval", TimeValue(Seconds(1.0 / 10000)));
    clientApps.Add(ulClientHelper.Install(groundNode));

    nrHelper->AttachToClosestGnb(groundNodeNetDev, gnbNetDev);

    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Stop(simTime);

    // Schedule UpdateAntennaOrientation for satellite nodes
    for (uint32_t i = 0; i < gnbNetDev.GetN(); i++)
    {
        auto satNetDevice = gnbNetDev.Get(i);
        auto gnbNetDevice = satNetDevice->GetObject<NrGnbNetDevice>();
        auto phy = gnbNetDevice->GetPhy(0);
        auto spectrumPhy = phy->GetSpectrumPhy();
        auto satAntenna = spectrumPhy->GetAntenna()->GetObject<UniformPlanarArray>();
        Ptr<Node> satNode = satNetDevice->GetNode();
        satNode->AggregateObject(satAntenna);
        UpdateAntennaOrientation(satNode, satAntenna, MilliSeconds(precision));
    }

    // FlowMonitor setup
    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(groundNodeContainer);

    Ptr<ns3::FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    monitor->SetAttribute("DelayBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("JitterBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));

    Simulator::Stop(simTime);

    double snapshotInterval = 0.1;
    std::map<FlowId, FlowMonitor::FlowStats> lastStats;
    std::ofstream traceFile("kpi_trace_scenario2_ntngap.csv");
    traceFile << "time,flowId,throughput_mbps,delay_ms,jitter_ms,lost_packets,sinr_db,rsrp_dbm,packet_loss_rate\n";
    Simulator::Schedule(Seconds(snapshotInterval), &SampleFlowStats, monitor, classifier,
                         std::ref(lastStats), std::ref(traceFile), snapshotInterval);

    Simulator::Run();

    // Print summary
    monitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
    double flowDuration = (simTime - udpAppStartTime).GetSeconds();
    for (auto i = stats.begin(); i != stats.end(); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        std::cout << "Flow " << i->first << " (" << t.sourceAddress << " -> "
                  << t.destinationAddress << ")";
        if (i->second.rxPackets > 0)
        {
            std::cout << " Throughput: " << i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000
                      << " Mbps";
        }
        std::cout << "\n";
    }

    Simulator::Destroy();
    return EXIT_SUCCESS;
}
