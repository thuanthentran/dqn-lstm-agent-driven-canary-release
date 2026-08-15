// Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

/**
 * @ingroup examples
 * @file cttc-nr-scenario-handover-storm.cc
 * @brief Scenario 1 (HandoverStorm): multiple gNBs with frequent scripted handovers.
 *
 * Based on cttc-nr-demo.cc baseline with handover patterns inspired by
 * nr-test-x2-handover.cc. Uses NrNoOpHandoverAlgorithm with manually
 * scheduled HandoverRequest calls to create a "handover storm" scenario.
 * UEs are teleported near the target gNB before each handover event.
 */

// NOLINTBEGIN
// clang-format off
// clang-format on
// NOLINTEND

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("CttcNrScenarioHandoverStorm");

// Global handover counter
static uint32_t g_handoverCount = 0;

/**
 * Callback for HandoverStart trace source.
 * @param context the context string
 * @param imsi the IMSI
 * @param cellId the cell ID
 * @param rnti the RNTI
 * @param targetCellId the target cell ID
 */
void
NotifyHandoverStartGnb(std::string context,
                       uint64_t imsi,
                       uint16_t cellId,
                       uint16_t rnti,
                       uint16_t targetCellId)
{
    g_handoverCount++;
    std::cout << Simulator::Now().GetSeconds() << "s: Handover #" << g_handoverCount
              << " IMSI=" << imsi << " from cellId=" << cellId
              << " to cellId=" << targetCellId << std::endl;
}

/**
 * Teleport UE near a target gNB node.
 * @param ueNode the UE node
 * @param gnbNode the target gNB node
 */
void
TeleportUeNearGnb(Ptr<Node> ueNode, Ptr<Node> gnbNode)
{
    Ptr<MobilityModel> gnbMob = gnbNode->GetObject<MobilityModel>();
    Vector pos = gnbMob->GetPosition();
    Ptr<MobilityModel> ueMob = ueNode->GetObject<MobilityModel>();
    ueMob->SetPosition(Vector(pos.x, pos.y + 5.0, 1.5));
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

        traceFile << now << "," << id << "," << throughputMbps << ","
                   << meanDelayMs << "," << meanJitterMs << "," << dLost << "\n";
    }
    lastStats = stats;

    Simulator::Schedule(Seconds(interval), &SampleFlowStats, monitor, classifier,
                         std::ref(lastStats), std::ref(traceFile), interval);
}

int
main(int argc, char* argv[])
{
    // Scenario parameters
    uint16_t gNbNum = 3;
    uint16_t ueNumPergNb = 2;
    bool logging = false;
    bool doubleOperationalBand = true;

    // Traffic parameters
    uint32_t udpPacketSizeULL = 100;
    uint32_t udpPacketSizeBe = 1252;
    uint32_t lambdaULL = 10000;
    uint32_t lambdaBe = 10000;

    // Simulation parameters — identical across all 5 scenarios
    Time simTime = MilliSeconds(1000);
    Time udpAppStartTime = MilliSeconds(400);

    // NR parameters
    uint16_t numerologyBwp1 = 4;
    double centralFrequencyBand1 = 28e9;
    double bandwidthBand1 = 50e6;
    uint16_t numerologyBwp2 = 2;
    double centralFrequencyBand2 = 28.2e9;
    double bandwidthBand2 = 50e6;
    double totalTxPower = 35;

    std::string simTag = "default";
    std::string outputDir = "./";

    CommandLine cmd(__FILE__);
    cmd.AddValue("gNbNum", "The number of gNbs in multiple-ue topology", gNbNum);
    cmd.AddValue("ueNumPergNb", "The number of UE per gNb in multiple-ue topology", ueNumPergNb);
    cmd.AddValue("logging", "Enable logging", logging);
    cmd.AddValue("doubleOperationalBand",
                 "If true, simulate two operational bands with one CC for each band,"
                 "and each CC will have 1 BWP that spans the entire CC.",
                 doubleOperationalBand);
    cmd.AddValue("packetSizeUll",
                 "packet size in bytes to be used by ultra low latency traffic",
                 udpPacketSizeULL);
    cmd.AddValue("packetSizeBe",
                 "packet size in bytes to be used by best effort traffic",
                 udpPacketSizeBe);
    cmd.AddValue("lambdaUll",
                 "Number of UDP packets in one second for ultra low latency traffic",
                 lambdaULL);
    cmd.AddValue("lambdaBe",
                 "Number of UDP packets in one second for best effort traffic",
                 lambdaBe);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.AddValue("numerologyBwp1", "The numerology to be used in bandwidth part 1", numerologyBwp1);
    cmd.AddValue("centralFrequencyBand1",
                 "The system frequency to be used in band 1",
                 centralFrequencyBand1);
    cmd.AddValue("bandwidthBand1", "The system bandwidth to be used in band 1", bandwidthBand1);
    cmd.AddValue("numerologyBwp2", "The numerology to be used in bandwidth part 2", numerologyBwp2);
    cmd.AddValue("centralFrequencyBand2",
                 "The system frequency to be used in band 2",
                 centralFrequencyBand2);
    cmd.AddValue("bandwidthBand2", "The system bandwidth to be used in band 2", bandwidthBand2);
    cmd.AddValue("totalTxPower",
                 "total tx power that will be proportionally assigned to"
                 " bands, CCs and bandwidth parts depending on each BWP bandwidth ",
                 totalTxPower);
    cmd.AddValue("simTag",
                 "tag to be appended to output filenames to distinguish simulation campaigns",
                 simTag);
    cmd.AddValue("outputDir", "directory where to store simulation results", outputDir);
    cmd.Parse(argc, argv);

    NS_ABORT_IF(centralFrequencyBand1 < 0.5e9 && centralFrequencyBand1 > 100e9);
    NS_ABORT_IF(centralFrequencyBand2 < 0.5e9 && centralFrequencyBand2 > 100e9);

    if (logging)
    {
        LogComponentEnable("UdpClient", LOG_LEVEL_INFO);
        LogComponentEnable("UdpServer", LOG_LEVEL_INFO);
        LogComponentEnable("NrPdcp", LOG_LEVEL_INFO);
    }

    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

    // Disable RLF detection to prevent premature RLF during handover
    // (same approach as nr-test-x2-handover.cc)
    Config::SetDefault("ns3::NrUePhy::EnableUplinkPowerControl", BooleanValue(false));

    /*
     * Create the scenario with 3 gNBs placed in a line, 500m apart.
     * UEs are placed near gNB 0 initially.
     */
    NodeContainer gnbNodes;
    gnbNodes.Create(gNbNum);
    NodeContainer ueNodes;
    ueNodes.Create(ueNumPergNb * gNbNum);

    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    // gNB positions: 500m apart in a line
    for (uint16_t i = 0; i < gNbNum; i++)
    {
        positionAlloc->Add(Vector(i * 500.0, 0, 10.0));
    }
    // UE positions: all start near gNB 0
    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        positionAlloc->Add(Vector(5.0, 5.0 + i * 2.0, 1.5));
    }
    MobilityHelper mobility;
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(gnbNodes);
    mobility.Install(ueNodes);

    // Separate UEs into lowLat and voice containers (same as baseline)
    NodeContainer ueLowLatContainer;
    NodeContainer ueVoiceContainer;
    for (uint32_t j = 0; j < ueNodes.GetN(); ++j)
    {
        if (j % 2 == 0)
        {
            ueLowLatContainer.Add(ueNodes.Get(j));
        }
        else
        {
            ueVoiceContainer.Add(ueNodes.Get(j));
        }
    }

    NS_LOG_INFO("Creating " << ueNodes.GetN() << " user terminals and "
                             << gnbNodes.GetN() << " gNBs");

    /*
     * Setup NR module with handover disabled (NoOp algorithm).
     * Handovers will be triggered manually via HandoverRequest.
     */
    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);

    // Use NoOp handover algorithm — handovers triggered manually
    nrHelper->SetHandoverAlgorithmType("ns3::NrNoOpHandoverAlgorithm");

    /*
     * Spectrum configuration — same as baseline but applied per-gNB.
     * All gNBs share the same band configuration.
     */
    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;

    CcBwpCreator::SimpleOperationBandConf bandConf1(centralFrequencyBand1,
                                                    bandwidthBand1,
                                                    numCcPerBand);
    OperationBandInfo band1 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf1);

    CcBwpCreator::SimpleOperationBandConf bandConf2(centralFrequencyBand2,
                                                    bandwidthBand2,
                                                    numCcPerBand);
    OperationBandInfo band2 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf2);

    double x = pow(10, totalTxPower / 10);
    double totalBandwidth = bandwidthBand1;

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    // Use Friis propagation for handover scenario (same as nr-test-x2-handover.cc)
    channelHelper->ConfigurePropagationFactory(FriisPropagationLossModel::GetTypeId());

    if (doubleOperationalBand)
    {
        channelHelper->AssignChannelsToBands({band1, band2});
        totalBandwidth += bandwidthBand2;
        allBwps = CcBwpCreator::GetAllBwps({band1, band2});
    }
    else
    {
        channelHelper->AssignChannelsToBands({band1});
        allBwps = CcBwpCreator::GetAllBwps({band1});
    }

    Packet::EnableChecking();
    Packet::EnablePrinting();

    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));
    nrEpcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    // Antennas for all the UEs (same as baseline)
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    // Antennas for all the gNbs (same as baseline)
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    uint32_t bwpIdForLowLat = 0;
    uint32_t bwpIdForVoice = 0;
    if (doubleOperationalBand)
    {
        bwpIdForVoice = 1;
        bwpIdForLowLat = 0;
    }

    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB",
                                                 UintegerValue(bwpIdForLowLat));
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(bwpIdForVoice));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(bwpIdForLowLat));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(bwpIdForVoice));

    /*
     * Install gNB devices. Each gNB gets its own band set (same frequencies
     * but separate channel instances, following the pattern from nr-test-x2-handover.cc
     * which installs each gNB with its own band).
     */
    NetDeviceContainer gnbNetDev;
    for (uint32_t i = 0; i < gnbNodes.GetN(); i++)
    {
        gnbNetDev.Add(nrHelper->InstallGnbDevice(gnbNodes.Get(i), allBwps));
    }

    // Install UE devices — provide BWPs from all bands for handover capability
    NetDeviceContainer ueLowLatNetDev = nrHelper->InstallUeDevice(ueLowLatContainer, allBwps);
    NetDeviceContainer ueVoiceNetDev = nrHelper->InstallUeDevice(ueVoiceContainer, allBwps);

    NetDeviceContainer ueNetDevs;
    ueNetDevs.Add(ueLowLatNetDev);
    ueNetDevs.Add(ueVoiceNetDev);
    nrHelper->AssignStreams({.gnbDevs = gnbNetDev, .ueDevs = ueNetDevs});

    // Set per-gNB PHY attributes
    for (uint32_t i = 0; i < gnbNetDev.GetN(); i++)
    {
        NrHelper::GetGnbPhy(gnbNetDev.Get(i), 0)
            ->SetAttribute("Numerology", UintegerValue(numerologyBwp1));
        NrHelper::GetGnbPhy(gnbNetDev.Get(i), 0)
            ->SetAttribute("TxPower", DoubleValue(10 * log10((bandwidthBand1 / totalBandwidth) * x)));

        if (doubleOperationalBand)
        {
            NrHelper::GetGnbPhy(gnbNetDev.Get(i), 1)
                ->SetAttribute("Numerology", UintegerValue(numerologyBwp2));
            NrHelper::GetGnbPhy(gnbNetDev.Get(i), 1)
                ->SetTxPower(10 * log10((bandwidthBand2 / totalBandwidth) * x));
        }
    }

    // Setup IP stack
    auto [remoteHost, remoteHostIpv4Address] =
        nrEpcHelper->SetupRemoteHost("100Gb/s", 2500, Seconds(0.000));

    InternetStackHelper internet;
    internet.Install(ueNodes);

    Ipv4InterfaceContainer ueLowLatIpIface =
        nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLowLatNetDev));
    Ipv4InterfaceContainer ueVoiceIpIface =
        nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueVoiceNetDev));

    // Attach all UEs to gNB 0 initially
    for (uint32_t i = 0; i < ueLowLatNetDev.GetN(); i++)
    {
        nrHelper->AttachToGnb(ueLowLatNetDev.Get(i), gnbNetDev.Get(0));
    }
    for (uint32_t i = 0; i < ueVoiceNetDev.GetN(); i++)
    {
        nrHelper->AttachToGnb(ueVoiceNetDev.Get(i), gnbNetDev.Get(0));
    }

    // Setup X2 interface between all gNBs (required for handover)
    nrHelper->AddX2Interface(gnbNodes);

    /*
     * Traffic configuration — same as baseline
     */
    uint16_t dlPortLowLat = 1234;
    uint16_t dlPortVoice = 1235;

    ApplicationContainer serverApps;
    UdpServerHelper dlPacketSinkLowLat(dlPortLowLat);
    UdpServerHelper dlPacketSinkVoice(dlPortVoice);
    serverApps.Add(dlPacketSinkLowLat.Install(ueLowLatContainer));
    serverApps.Add(dlPacketSinkVoice.Install(ueVoiceContainer));

    UdpClientHelper dlClientLowLat;
    dlClientLowLat.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClientLowLat.SetAttribute("PacketSize", UintegerValue(udpPacketSizeULL));
    dlClientLowLat.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambdaULL)));

    NrQosFlow lowLatFlow(NrQosFlow::NGBR_LOW_LAT_EMBB);
    Ptr<NrQosRule> lowLatRule = Create<NrQosRule>();
    NrQosRule::PacketFilter dlpfLowLat;
    dlpfLowLat.localPortStart = dlPortLowLat;
    dlpfLowLat.localPortEnd = dlPortLowLat;
    lowLatRule->Add(dlpfLowLat);

    UdpClientHelper dlClientVoice;
    dlClientVoice.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClientVoice.SetAttribute("PacketSize", UintegerValue(udpPacketSizeBe));
    dlClientVoice.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambdaBe)));

    NrQosFlow voiceFlow(NrQosFlow::GBR_CONV_VOICE);
    Ptr<NrQosRule> voiceRule = Create<NrQosRule>();
    NrQosRule::PacketFilter dlpfVoice;
    dlpfVoice.localPortStart = dlPortVoice;
    dlpfVoice.localPortEnd = dlPortVoice;
    voiceRule->Add(dlpfVoice);

    ApplicationContainer clientApps;

    for (uint32_t i = 0; i < ueLowLatContainer.GetN(); ++i)
    {
        Ptr<NetDevice> ueDevice = ueLowLatNetDev.Get(i);
        Address ueAddress = ueLowLatIpIface.GetAddress(i);

        dlClientLowLat.SetAttribute(
            "Remote",
            AddressValue(addressUtils::ConvertToSocketAddress(ueAddress, dlPortLowLat)));
        clientApps.Add(dlClientLowLat.Install(remoteHost));
        nrHelper->ActivateDedicatedQosFlow(ueDevice, lowLatFlow, lowLatRule);
    }

    for (uint32_t i = 0; i < ueVoiceContainer.GetN(); ++i)
    {
        Ptr<NetDevice> ueDevice = ueVoiceNetDev.Get(i);
        Address ueAddress = ueVoiceIpIface.GetAddress(i);

        dlClientVoice.SetAttribute(
            "Remote",
            AddressValue(addressUtils::ConvertToSocketAddress(ueAddress, dlPortVoice)));
        clientApps.Add(dlClientVoice.Install(remoteHost));
        nrHelper->ActivateDedicatedQosFlow(ueDevice, voiceFlow, voiceRule);
    }

    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Stop(simTime);

    /*
     * Schedule handover events: ping-pong UE 0 between gNB 0, 1, 2 and back.
     * We target ~6 handover events across the 1s simulation.
     * The first UE (lowLat UE 0) is used for handovers.
     */
    // HO 1: t=450ms, UE0 from gNB0 -> gNB1
    Simulator::Schedule(MilliSeconds(440), &TeleportUeNearGnb, ueNodes.Get(0), gnbNodes.Get(1));
    nrHelper->HandoverRequest(MilliSeconds(450), ueLowLatNetDev.Get(0),
                              gnbNetDev.Get(0), gnbNetDev.Get(1));

    // HO 2: t=550ms, UE0 from gNB1 -> gNB2
    Simulator::Schedule(MilliSeconds(540), &TeleportUeNearGnb, ueNodes.Get(0), gnbNodes.Get(2));
    nrHelper->HandoverRequest(MilliSeconds(550), ueLowLatNetDev.Get(0),
                              gnbNetDev.Get(1), gnbNetDev.Get(2));

    // HO 3: t=650ms, UE0 from gNB2 -> gNB0
    Simulator::Schedule(MilliSeconds(640), &TeleportUeNearGnb, ueNodes.Get(0), gnbNodes.Get(0));
    nrHelper->HandoverRequest(MilliSeconds(650), ueLowLatNetDev.Get(0),
                              gnbNetDev.Get(2), gnbNetDev.Get(0));

    // HO 4: t=750ms, UE0 from gNB0 -> gNB1
    Simulator::Schedule(MilliSeconds(740), &TeleportUeNearGnb, ueNodes.Get(0), gnbNodes.Get(1));
    nrHelper->HandoverRequest(MilliSeconds(750), ueLowLatNetDev.Get(0),
                              gnbNetDev.Get(0), gnbNetDev.Get(1));

    // HO 5: t=850ms, UE0 from gNB1 -> gNB2
    Simulator::Schedule(MilliSeconds(840), &TeleportUeNearGnb, ueNodes.Get(0), gnbNodes.Get(2));
    nrHelper->HandoverRequest(MilliSeconds(850), ueLowLatNetDev.Get(0),
                              gnbNetDev.Get(1), gnbNetDev.Get(2));

    // HO 6: t=950ms, UE0 from gNB2 -> gNB0
    Simulator::Schedule(MilliSeconds(940), &TeleportUeNearGnb, ueNodes.Get(0), gnbNodes.Get(0));
    nrHelper->HandoverRequest(MilliSeconds(950), ueLowLatNetDev.Get(0),
                              gnbNetDev.Get(2), gnbNetDev.Get(0));

    // Connect HandoverStart trace source to count handovers
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::NrGnbNetDevice/NrGnbRrc/HandoverStart",
                    MakeCallback(&NotifyHandoverStartGnb));

    // FlowMonitor setup
    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(ueNodes);

    Ptr<ns3::FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    monitor->SetAttribute("DelayBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("JitterBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));

    Simulator::Stop(simTime);

    double snapshotInterval = 0.1;
    std::map<FlowId, FlowMonitor::FlowStats> lastStats;
    std::ofstream traceFile("kpi_trace_scenario1_handoverstorm.csv");
    traceFile << "time,flowId,throughput_mbps,delay_ms,jitter_ms,lost_packets\n";
    Simulator::Schedule(Seconds(snapshotInterval), &SampleFlowStats, monitor, classifier,
                         std::ref(lastStats), std::ref(traceFile), snapshotInterval);

    Simulator::Run();

    // Print per-flow statistics
    monitor->CheckForLostPackets();
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    double averageFlowThroughput = 0.0;
    double averageFlowDelay = 0.0;

    std::ofstream outFile;
    std::string filename = outputDir + "/" + simTag;
    outFile.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (!outFile.is_open())
    {
        std::cerr << "Can't open file " << filename << std::endl;
        return 1;
    }

    outFile.setf(std::ios_base::fixed);

    double flowDuration = (simTime - udpAppStartTime).GetSeconds();
    for (auto i = stats.begin(); i != stats.end(); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        std::stringstream protoStream;
        protoStream << (uint16_t)t.protocol;
        if (t.protocol == 6)
        {
            protoStream.str("TCP");
        }
        if (t.protocol == 17)
        {
            protoStream.str("UDP");
        }
        outFile << "Flow " << i->first << " (" << t.sourceAddress << ":" << t.sourcePort << " -> "
                << t.destinationAddress << ":" << t.destinationPort << ") proto "
                << protoStream.str() << "\n";
        outFile << "  Tx Packets: " << i->second.txPackets << "\n";
        outFile << "  Tx Bytes:   " << i->second.txBytes << "\n";
        outFile << "  TxOffered:  " << i->second.txBytes * 8.0 / flowDuration / 1000.0 / 1000.0
                << " Mbps\n";
        outFile << "  Rx Bytes:   " << i->second.rxBytes << "\n";
        if (i->second.rxPackets > 0)
        {
            averageFlowThroughput += i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000;
            averageFlowDelay += 1000 * i->second.delaySum.GetSeconds() / i->second.rxPackets;

            outFile << "  Throughput: " << i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000
                    << " Mbps\n";
            outFile << "  Mean delay:  "
                    << 1000 * i->second.delaySum.GetSeconds() / i->second.rxPackets << " ms\n";
            outFile << "  Mean jitter:  "
                    << 1000 * i->second.jitterSum.GetSeconds() / i->second.rxPackets << " ms\n";
        }
        else
        {
            outFile << "  Throughput:  0 Mbps\n";
            outFile << "  Mean delay:  0 ms\n";
            outFile << "  Mean jitter: 0 ms\n";
        }
        outFile << "  Rx Packets: " << i->second.rxPackets << "\n";
    }

    double meanFlowThroughput = averageFlowThroughput / stats.size();
    double meanFlowDelay = averageFlowDelay / stats.size();

    outFile << "\n\n  Mean flow throughput: " << meanFlowThroughput << "\n";
    outFile << "  Mean flow delay: " << meanFlowDelay << "\n";

    outFile.close();

    std::ifstream f(filename.c_str());
    if (f.is_open())
    {
        std::cout << f.rdbuf();
    }

    std::cout << "\n=== Handover Storm Summary ===\n";
    std::cout << "Total handover events observed: " << g_handoverCount << "\n";

    Simulator::Destroy();
    return EXIT_SUCCESS;
}
