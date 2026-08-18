// Copyright (c) 2019 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
//
// SPDX-License-Identifier: GPL-2.0-only

/**
 * @ingroup examples
 * @file cttc-nr-scenario-thz-blockage.cc
 * @brief Scenario 3 (THzBlockage): periodic TxPower reduction to simulate blockage windows.
 *
 * Approximates THz/mmWave blockage by periodically dropping gNB TxPower
 * to a very low value, simulating severe signal degradation. The blockage
 * pattern follows step % 15 < 3 cadence (blockage on for 3 steps, off for 12).
 *
 * Notice: this entire program uses technical terms defined by the 3GPP TS 38.300 [1].
 *
 * This example describes how to setup a simulation using the 3GPP channel model from TR 38.901 [2].
 * This example consists of a simple grid topology, in which you
 * can choose the number of gNbs and UEs. Have a look at the possible parameters
 * to know what you can configure through the command line.
 *
 * With the default configuration, the example will create two flows that will
 * go through two different subband numerologies (or bandwidth parts). For that,
 * specifically, two bands are created, each with a single CC, and each CC containing
 * one bandwidth part.
 *
 * The example will print on-screen the end-to-end result of one (or two) flows,
 * as well as writing them on a file.
 *
 * \code{.unparsed}
$ ./ns3 run "cttc-nr-demo --PrintHelp"
    \endcode
 *
 */

// NOLINTBEGIN
// clang-format off

/**
 * Useful references that will be used for this tutorial:
 * [1] <a href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3191">3GPP TS 38.300</a>
 * [2] <a href="https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173">3GPP channel model from TR 38.901</a>
 * [3] <a href="https://www.nsnam.org/docs/release/3.38/tutorial/html/tweaking.html#using-the-logging-module">ns-3 documentation</a>
 */

// clang-format on
// NOLINTEND

/*
 * Include part. Often, you will have to include the headers for an entire module;
 * do that by including the name of the module you need with the suffix "-module.h".
 */

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

/*
 * Use, always, the namespace ns3. All the NR classes are inside such namespace.
 */
using namespace ns3;

/*
 * With this line, we will be able to see the logs of the file by enabling the
 * component "CttcNrDemo".
 * Further information on how logging works can be found in the ns-3 documentation [3].
 */
NS_LOG_COMPONENT_DEFINE("CttcNrScenarioThzBlockage");

// Store the original TxPower values to restore after blockage
static double g_originalTxPowerBwp0 = 0.0;
static double g_originalTxPowerBwp1 = 0.0;
static bool g_blockageActive = false;
static uint32_t g_blockageStep = 0;

// Global PHY-layer telemetry
static double g_lastSinrDb = 20.0;
static double g_lastRsrpDbm = -75.0;

void RecordDlDataSinr(uint16_t cellId, uint16_t rnti, double sinr, uint16_t ccId) {
    g_lastSinrDb = (sinr > 0) ? 10.0 * std::log10(sinr) : -30.0;
}

void RecordReportRsrp(uint16_t p1, uint16_t p2, uint16_t p3, double rsrp, uint8_t ccId) {
    g_lastRsrpDbm = (rsrp > 0) ? 10.0 * std::log10(rsrp * 1000.0) : -140.0;
}


/**
 * Toggle blockage on/off following step % 15 < 3 pattern.
 * During blockage, gNB TxPower is dropped to 5 dBm.
 * @param gnbNetDev the gNB net device container
 * @param doubleOp whether double operational band is used
 * @param toggleInterval interval between blockage checks
 */
void
ToggleBlockage(NetDeviceContainer gnbNetDev, bool doubleOp, double toggleInterval)
{
    // Blockage pattern: 2 steps ON, 5 steps OFF (cycle of 7).
    // This ensures multiple blockage windows during the traffic period (0.4-1.0s).
    // With 0.1s intervals: blockage at steps 0-1 (0.4-0.5s), 7-8 (1.1-1.2s), etc.
    bool shouldBlock = (g_blockageStep % 7) < 2;

    if (shouldBlock && !g_blockageActive)
    {
        // Enter blockage: reduce TxPower by 60 dB (simulating human/vehicle blockage per 3GPP TR 38.901)
        NrHelper::GetGnbPhy(gnbNetDev.Get(0), 0)->SetTxPower(g_originalTxPowerBwp0 - 60.0);
        if (doubleOp)
        {
            NrHelper::GetGnbPhy(gnbNetDev.Get(0), 1)->SetTxPower(g_originalTxPowerBwp1 - 60.0);
        }
        g_blockageActive = true;
        NS_LOG_INFO(Simulator::Now().GetSeconds() << "s: BLOCKAGE ON (step " << g_blockageStep << ")");
    }
    else if (!shouldBlock && g_blockageActive)
    {
        // Exit blockage: restore original TxPower
        NrHelper::GetGnbPhy(gnbNetDev.Get(0), 0)->SetTxPower(g_originalTxPowerBwp0);
        if (doubleOp)
        {
            NrHelper::GetGnbPhy(gnbNetDev.Get(0), 1)->SetTxPower(g_originalTxPowerBwp1);
        }
        g_blockageActive = false;
        NS_LOG_INFO(Simulator::Now().GetSeconds() << "s: BLOCKAGE OFF (step " << g_blockageStep << ")");
    }

    g_blockageStep++;
    Simulator::Schedule(Seconds(toggleInterval), &ToggleBlockage, gnbNetDev, doubleOp, toggleInterval);
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
int
main(int argc, char* argv[])
{
    /*
     * Variables that represent the parameters we will accept as input by the
     * command line. Each of them is initialized with a default value, and
     * possibly overridden below when command-line arguments are parsed.
     */
    // Scenario parameters (that we will use inside this script):
    uint16_t gNbNum = 1;
    uint16_t ueNumPergNb = 2;
    bool logging = false;
    bool doubleOperationalBand = true;

    // Traffic parameters (that we will use inside this script):
    uint32_t udpPacketSizeULL = 100;
    uint32_t udpPacketSizeBe = 1252;
    uint32_t lambdaULL = 10000;
    uint32_t lambdaBe = 10000;

    // Simulation parameters. Please don't use double to indicate seconds; use
    // ns-3 Time values which use integers to avoid portability issues.
    Time simTime = MilliSeconds(30000);
    Time udpAppStartTime = MilliSeconds(400);

    // NR parameters (Reference: 3GPP TR 38.901 V17.0.0 (Release 17)
    // Table 7.8-1 for the power and BW).
    // In this example the BW has been split into two BWPs
    // We will take the input from the command line, and then we
    // will pass them inside the NR module.
    uint16_t numerologyBwp1 = 4;
    double centralFrequencyBand1 = 28e9;
    double bandwidthBand1 = 50e6;
    uint16_t numerologyBwp2 = 2;
    double centralFrequencyBand2 = 28.2e9;
    double bandwidthBand2 = 50e6;
    double totalTxPower = 35;

    // Where we will store the output files.
    std::string simTag = "default";
    std::string outputDir = "./";

    /*
     * From here, we instruct the ns3::CommandLine class of all the input parameters
     * that we may accept as input, as well as their description, and the storage
     * variable.
     */
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

    // Parse the command line
    cmd.Parse(argc, argv);

    /*
     * Check if the frequency is in the allowed range.
     * If you need to add other checks, here is the best position to put them.
     */
    NS_ABORT_IF(centralFrequencyBand1 < 0.5e9 && centralFrequencyBand1 > 100e9);
    NS_ABORT_IF(centralFrequencyBand2 < 0.5e9 && centralFrequencyBand2 > 100e9);

    /*
     * If the logging variable is set to true, enable the log of some components
     * through the code. The same effect can be obtained through the use
     * of the NS_LOG environment variable:
     *
     * export NS_LOG="UdpClient=level_info|prefix_time|prefix_func|prefix_node:UdpServer=..."
     *
     * Usually, the environment variable way is preferred, as it is more customizable,
     * and more expressive.
     */
    if (logging)
    {
        LogComponentEnable("UdpClient", LOG_LEVEL_INFO);
        LogComponentEnable("UdpServer", LOG_LEVEL_INFO);
        LogComponentEnable("NrPdcp", LOG_LEVEL_INFO);
    }

    /*
     * In general, attributes for the NR module are typically configured in NrHelper.  However, some
     * attributes need to be configured globally through the Config::SetDefault() method. Below is
     * an example: if you want to make the RLC buffer very large, you can pass a very large integer
     * here.
     */
    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

    /*
     * Create the scenario. In our examples, we heavily use helpers that setup
     * the gnbs and ue following a pre-defined pattern. Please have a look at the
     * GridScenarioHelper documentation to see how the nodes will be distributed.
     */
    GridScenarioHelper gridScenario;
    gridScenario.SetRows(1);
    gridScenario.SetColumns(gNbNum);
    // All units below are in meters
    gridScenario.SetHorizontalBsDistance(10.0);
    gridScenario.SetVerticalBsDistance(10.0);
    gridScenario.SetBsHeight(10);
    gridScenario.SetUtHeight(1.5);
    // must be set before BS number
    gridScenario.SetSectorization(GridScenarioHelper::SINGLE);
    gridScenario.SetBsNumber(gNbNum);
    gridScenario.SetUtNumber(ueNumPergNb * gNbNum);
    gridScenario.SetScenarioHeight(3); // Create a 3x3 scenario where the UE will
    gridScenario.SetScenarioLength(3); // be distributed.
    gridScenario.CreateScenario();

    /*
     * Create two different NodeContainer for the different traffic type.
     * In ueLowLat we will put the UEs that will receive low-latency traffic,
     * while in ueVoice we will put the UEs that will receive the voice traffic.
     */
    NodeContainer ueLowLatContainer;
    NodeContainer ueVoiceContainer;

    for (uint32_t j = 0; j < gridScenario.GetUserTerminals().GetN(); ++j)
    {
        Ptr<Node> ue = gridScenario.GetUserTerminals().Get(j);
        if (j % 2 == 0)
        {
            ueLowLatContainer.Add(ue);
        }
        else
        {
            ueVoiceContainer.Add(ue);
        }
    }

    /*
     * TODO: Add a print, or a plot, that shows the scenario.
     */
    NS_LOG_INFO("Creating " << gridScenario.GetUserTerminals().GetN() << " user terminals and "
                            << gridScenario.GetBaseStations().GetN() << " gNBs");

    /*
     * Setup the NR module. We create the various helpers needed for the
     * NR simulation:
     * - nrEpcHelper, which will setup the core network
     * - IdealBeamformingHelper, which takes care of the beamforming part
     * - NrHelper, which takes care of creating and connecting the various
     * part of the NR stack
     * - NrChannelHelper, which takes care of the spectrum channel
     */
    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    // Put the pointers inside nrHelper
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);

    /*
     * Spectrum division. We create two operational bands, each of them containing
     * one component carrier, and each CC containing a single bandwidth part
     * centered at the frequency specified by the input parameters.
     * Each spectrum part length is, as well, specified by the input parameters.
     * Both operational bands will use the StreetCanyon channel modeling.
     */
    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1; // in this example, both bands have a single CC

    // Create the configuration for the CcBwpHelper. SimpleOperationBandConf creates
    // a single BWP per CC
    CcBwpCreator::SimpleOperationBandConf bandConf1(centralFrequencyBand1,
                                                    bandwidthBand1,
                                                    numCcPerBand);

    // Create the band and install the channel into it
    OperationBandInfo band1 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf1);
    // Set the channel for the band
    CcBwpCreator::SimpleOperationBandConf bandConf2(centralFrequencyBand2,
                                                    bandwidthBand2,
                                                    numCcPerBand);
    OperationBandInfo band2 = ccBwpCreator.CreateOperationBandContiguousCc(bandConf2);

    /*
     * The configured spectrum division is:
     * ------------Band1--------------|--------------Band2-----------------
     * ------------CC1----------------|--------------CC2-------------------
     * ------------BWP1---------------|--------------BWP2------------------
     */

    /*
     * Start to account for the bandwidth used by the example, as well as
     * the total power that has to be divided among the BWPs.
     */
    double x = pow(10, totalTxPower / 10);
    double totalBandwidth = bandwidthBand1;
    /**
     * The channel is configured by this helper using a combination of the scenario, the channel
     * condition model, and the fading model.
     */

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    /**
     * Use channelHelper API to define the attributes for the channel model (condition, pathloss and
     * spectrum)
     */
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    /*
     * if not single band simulation, initialize and setup power in the second band.
     * Install channel and pathloss, plus other things inside single or both bands.
     */
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

    /*
     * allBwps contains all the spectrum configuration needed for the nrHelper.
     *
     * Now, we can setup the attributes. We can have three kind of attributes:
     * (i) parameters that are valid for all the bandwidth parts and applies to
     * all nodes, (ii) parameters that are valid for all the bandwidth parts
     * and applies to some node only, and (iii) parameters that are different for
     * every bandwidth parts. The approach is:
     *
     * - for (i): Configure the attribute through the helper, and then install;
     * - for (ii): Configure the attribute through the helper, and then install
     * for the first set of nodes. Then, change the attribute through the helper,
     * and install again;
     * - for (iii): Install, and then configure the attributes by retrieving
     * the pointer needed, and calling "SetAttribute" on top of such pointer.
     *
     */

    Packet::EnableChecking();
    Packet::EnablePrinting();

    /*
     *  Case (i): Attributes valid for all the nodes
     */
    // Beamforming method
    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));

    // Core latency
    nrEpcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    // Antennas for all the UEs
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    // Antennas for all the gNbs
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

    // gNb routing between Bearer and bandwidh part
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB",
                                                 UintegerValue(bwpIdForLowLat));
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(bwpIdForVoice));

    // Ue routing between Bearer and bandwidth part
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(bwpIdForLowLat));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(bwpIdForVoice));

    /*
     * We miss many other parameters. By default, not configuring them is equivalent
     * to use the default values. Please, have a look at the documentation to see
     * what are the default values for all the attributes you are not seeing here.
     */

    /*
     * Case (ii): Attributes valid for a subset of the nodes
     */

    // NOT PRESENT IN THIS SIMPLE EXAMPLE

    /*
     * We have configured the attributes we needed. Now, install and get the pointers
     * to the NetDevices, which contains all the NR stack:
     */

    NetDeviceContainer gnbNetDev =
        nrHelper->InstallGnbDevice(gridScenario.GetBaseStations(), allBwps);
    NetDeviceContainer ueLowLatNetDev = nrHelper->InstallUeDevice(ueLowLatContainer, allBwps);
    NetDeviceContainer ueVoiceNetDev = nrHelper->InstallUeDevice(ueVoiceContainer, allBwps);

    NetDeviceContainer ueNetDevs;
    ueNetDevs.Add(ueLowLatNetDev);
    ueNetDevs.Add(ueVoiceNetDev);
    nrHelper->AssignStreams({.scenario = &gridScenario, .gnbDevs = gnbNetDev, .ueDevs = ueNetDevs});
    /*
     * Case (iii): Go node for node and change the attributes we have to setup
     * per-node.
     */

    // Get the first netdevice (gnbNetDev.Get (0)) and the first bandwidth part (0)
    // and set the attribute.
    NrHelper::GetGnbPhy(gnbNetDev.Get(0), 0)
        ->SetAttribute("Numerology", UintegerValue(numerologyBwp1));
    NrHelper::GetGnbPhy(gnbNetDev.Get(0), 0)
        ->SetAttribute("TxPower", DoubleValue(10 * log10((bandwidthBand1 / totalBandwidth) * x)));

    if (doubleOperationalBand)
    {
        // Get the first netdevice (gnbNetDev.Get (0)) and the second bandwidth part (1)
        // and set the attribute.
        NrHelper::GetGnbPhy(gnbNetDev.Get(0), 1)
            ->SetAttribute("Numerology", UintegerValue(numerologyBwp2));
        NrHelper::GetGnbPhy(gnbNetDev.Get(0), 1)
            ->SetTxPower(10 * log10((bandwidthBand2 / totalBandwidth) * x));
    }

    // From here, it is standard NS3. In the future, we will create helpers
    // for this part as well.

    auto [remoteHost, remoteHostIpv4Address] =
        nrEpcHelper->SetupRemoteHost("100Gb/s", 2500, Seconds(0.000));

    InternetStackHelper internet;

    internet.Install(gridScenario.GetUserTerminals());

    Ipv4InterfaceContainer ueLowLatIpIface =
        nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLowLatNetDev));
    Ipv4InterfaceContainer ueVoiceIpIface =
        nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(ueVoiceNetDev));

    // attach UEs to the closest gNB
    nrHelper->AttachToClosestGnb(ueLowLatNetDev, gnbNetDev);
    nrHelper->AttachToClosestGnb(ueVoiceNetDev, gnbNetDev);

    /*
     * Traffic part. Install two kind of traffic: low-latency and voice, each
     * identified by a particular source port.
     */
    uint16_t dlPortLowLat = 1234;
    uint16_t dlPortVoice = 1235;

    ApplicationContainer serverApps;

    // The sink will always listen to the specified ports
    UdpServerHelper dlPacketSinkLowLat(dlPortLowLat);
    UdpServerHelper dlPacketSinkVoice(dlPortVoice);

    // The server, that is the application which is listening, is installed in the UE
    serverApps.Add(dlPacketSinkLowLat.Install(ueLowLatContainer));
    serverApps.Add(dlPacketSinkVoice.Install(ueVoiceContainer));

    /*
     * Configure attributes for the different generators, using user-provided
     * parameters for generating a CBR traffic
     *
     * Low-Latency configuration and object creation:
     */
    UdpClientHelper dlClientLowLat;
    dlClientLowLat.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClientLowLat.SetAttribute("PacketSize", UintegerValue(udpPacketSizeULL));
    dlClientLowLat.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambdaULL)));

    // The bearer that will carry low latency traffic
    NrQosFlow lowLatFlow(NrQosFlow::NGBR_LOW_LAT_EMBB);

    // The filter for the low-latency traffic
    Ptr<NrQosRule> lowLatRule = Create<NrQosRule>();
    NrQosRule::PacketFilter dlpfLowLat;
    dlpfLowLat.localPortStart = dlPortLowLat;
    dlpfLowLat.localPortEnd = dlPortLowLat;
    lowLatRule->Add(dlpfLowLat);

    // Voice configuration and object creation:
    UdpClientHelper dlClientVoice;
    dlClientVoice.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClientVoice.SetAttribute("PacketSize", UintegerValue(udpPacketSizeBe));
    dlClientVoice.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambdaBe)));

    // The bearer that will carry voice traffic
    NrQosFlow voiceFlow(NrQosFlow::GBR_CONV_VOICE);

    // The filter for the voice traffic
    Ptr<NrQosRule> voiceRule = Create<NrQosRule>();
    NrQosRule::PacketFilter dlpfVoice;
    dlpfVoice.localPortStart = dlPortVoice;
    dlpfVoice.localPortEnd = dlPortVoice;
    voiceRule->Add(dlpfVoice);

    /*
     * Let's install the applications!
     */
    ApplicationContainer clientApps;

    for (uint32_t i = 0; i < ueLowLatContainer.GetN(); ++i)
    {
        Ptr<Node> ue = ueLowLatContainer.Get(i);
        Ptr<NetDevice> ueDevice = ueLowLatNetDev.Get(i);
        Address ueAddress = ueLowLatIpIface.GetAddress(i);

        // The client, who is transmitting, is installed in the remote host,
        // with destination address set to the address of the UE
        dlClientLowLat.SetAttribute(
            "Remote",
            AddressValue(addressUtils::ConvertToSocketAddress(ueAddress, dlPortLowLat)));
        clientApps.Add(dlClientLowLat.Install(remoteHost));

        // Activate a dedicated bearer for the traffic type
        nrHelper->ActivateDedicatedQosFlow(ueDevice, lowLatFlow, lowLatRule);
    }

    for (uint32_t i = 0; i < ueVoiceContainer.GetN(); ++i)
    {
        Ptr<Node> ue = ueVoiceContainer.Get(i);
        Ptr<NetDevice> ueDevice = ueVoiceNetDev.Get(i);
        Address ueAddress = ueVoiceIpIface.GetAddress(i);

        // The client, who is transmitting, is installed in the remote host,
        // with destination address set to the address of the UE
        dlClientVoice.SetAttribute(
            "Remote",
            AddressValue(addressUtils::ConvertToSocketAddress(ueAddress, dlPortVoice)));
        clientApps.Add(dlClientVoice.Install(remoteHost));

        // Activate a dedicated bearer for the traffic type
        nrHelper->ActivateDedicatedQosFlow(ueDevice, voiceFlow, voiceRule);
    }

    // start UDP server and client apps
    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Stop(simTime);

    // enable the traces provided by the nr module
    // nrHelper->EnableTraces();
    // --- PHY traces: SINR and RSRP (ns-3.48 / cttc-nr correct signatures) ---
    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*/$ns3::NrUeNetDevice/ComponentCarrierMapUe/*/NrUePhy/DlDataSinr",
        MakeCallback(&RecordDlDataSinr));

    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*/$ns3::NrUeNetDevice/ComponentCarrierMapUe/*/NrUePhy/ReportRsrp",
        MakeCallback(&RecordReportRsrp));


    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(gridScenario.GetUserTerminals());

    Ptr<ns3::FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    monitor->SetAttribute("DelayBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("JitterBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));

    Simulator::Stop(simTime);

    double snapshotInterval = 0.1;
    std::map<FlowId, FlowMonitor::FlowStats> lastStats;
    std::ofstream traceFile("kpi_trace_scenario3_thzblockage.csv");
    traceFile << "time,flowId,throughput_mbps,delay_ms,jitter_ms,lost_packets,sinr_db,rsrp_dbm,packet_loss_rate\n";
    Simulator::Schedule(Seconds(snapshotInterval), &SampleFlowStats, monitor, classifier,
                         std::ref(lastStats), std::ref(traceFile), snapshotInterval);

    // Store original TxPower values for blockage toggling
    g_originalTxPowerBwp0 = 10 * log10((bandwidthBand1 / totalBandwidth) * x);
    if (doubleOperationalBand)
    {
        g_originalTxPowerBwp1 = 10 * log10((bandwidthBand2 / totalBandwidth) * x);
    }

    // Schedule blockage toggling starting at udpAppStartTime so blockage
    // windows fall within the active traffic period.
    double blockageToggleInterval = snapshotInterval;
    Simulator::Schedule(udpAppStartTime, &ToggleBlockage, gnbNetDev, doubleOperationalBand, blockageToggleInterval);

    Simulator::Run();

    /*
     * To check what was installed in the memory, i.e., BWPs of gNB Device, and its configuration.
     * Example is: Node 1 -> Device 0 -> BandwidthPartMap -> {0,1} BWPs -> NrGnbPhy -> Numerology,
    GtkConfigStore config;
    config.ConfigureAttributes ();
    */

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
            // Measure the duration of the flow from receiver's perspective
            averageFlowThroughput += i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000;
            averageFlowDelay += 1000 * i->second.delaySum.GetSeconds() / i->second.rxPackets;

            outFile << "  Throughput: " << i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000
                    << " Mbps\n";
            outFile << "  Mean delay:  "
                    << 1000 * i->second.delaySum.GetSeconds() / i->second.rxPackets << " ms\n";
            // outFile << "  Mean upt:  " << i->second.uptSum / i->second.rxPackets / 1000/1000 << "
            // Mbps \n";
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

    Simulator::Destroy();

    if (argc == 0)
    {
        double toleranceMeanFlowThroughput = 0.0001 * 56.258560;
        double toleranceMeanFlowDelay = 0.0001 * 0.553292;

        if (meanFlowThroughput >= 56.258560 - toleranceMeanFlowThroughput &&
            meanFlowThroughput <= 56.258560 + toleranceMeanFlowThroughput &&
            meanFlowDelay >= 0.553292 - toleranceMeanFlowDelay &&
            meanFlowDelay <= 0.553292 + toleranceMeanFlowDelay)
        {
            return EXIT_SUCCESS;
        }
        else
        {
            return EXIT_FAILURE;
        }
    }
    else if (argc == 1 and ueNumPergNb == 9) // called from examples-to-run.py with these parameters
    {
        double toleranceMeanFlowThroughput = 0.0001 * 47.858536;
        double toleranceMeanFlowDelay = 0.0001 * 10.504189;

        if (meanFlowThroughput >= 47.858536 - toleranceMeanFlowThroughput &&
            meanFlowThroughput <= 47.858536 + toleranceMeanFlowThroughput &&
            meanFlowDelay >= 10.504189 - toleranceMeanFlowDelay &&
            meanFlowDelay <= 10.504189 + toleranceMeanFlowDelay)
        {
            return EXIT_SUCCESS;
        }
        else
        {
            return EXIT_FAILURE;
        }
    }
    else
    {
        return EXIT_SUCCESS; // we dont check other parameters configurations at the moment
    }
}
