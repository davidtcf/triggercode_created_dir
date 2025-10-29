#include "TriggerEfficiencyExtTriggerEvent.h"

TriggerEfficiencyExtTriggerEvent::TriggerEfficiencyExtTriggerEvent(EventBase* base)
    : tpcHGPulseArea(base->m_reader, "pulsesTPCHG.pulseArea_phd")
    , tpcnPulses(base->m_reader, "pulsesTPCHG.nPulses")// Number of TPC pulses for loops
    , tpcpulseStartTime_ns(base->m_reader, "pulsesTPCHG.pulseStartTime_ns")//TPC pulse start time for time correlating w/trigger
    , tpcpulseEndTime_ns(base->m_reader, "pulsesTPCHG.pulseEndTime_ns") // TPC pulse end time for time correlating w/trigger
    , trgSumnPulses(base->m_reader,"triggerSum.nPulses") // Number of trigger pulses for loops
    , trgSumpulseStartTime(base->m_reader,"triggerSum.pulseStartTime_ns") // Trigger Pulse Start 
    , trgSumCHID(base->m_reader,"triggerSum.channelID") //Trigger chID - 2106 -> S2 trigger
    , tpctotArea(base->m_reader,"tpcEventRQs.totalArea_phd") //total event area used for cut to avoid buffer fill
    , pulseClassification(base->m_reader,"pulsesTPCHG.classification") // Pulse Classifier to get S1, S2 info
    , chPulseArea_phd(base->m_reader,"pulsesTPCHG.chPulseArea_phd")//channel by channel pulse area to get max
    , s1PulseIDs(base->m_reader,"pulsesTPCHG.s1PulseIDs")
    , s2PulseIDs(base->m_reader,"pulsesTPCHG.s2PulseIDs")
    , sEPulseIDs(base->m_reader,"pulsesTPCHG.singleElectronPulseIDs")
    , otherPulseIDs(base->m_reader,"pulsesTPCHG.otherPulseIDs")
    , negativeArea(base->m_reader,"pulsesTPCHG.negativeArea_phd")
    , pulseArea200ns_phd(base->m_reader,"pulsesTPCHG.pulseArea200ns_phd")
    , s2XYchi2(base->m_reader,"pulsesTPC.s2XYchi2")
    , s2X_cm(base->m_reader,"pulsesTPC.s2Xposition_cm")
    , s2Y_cm(base->m_reader,"pulsesTPC.s2Yposition_cm")
    , aft10(base->m_reader,"pulsesTPCHG.areaFractionTime10_ns")
    , aft90(base->m_reader,"pulsesTPCHG.areaFractionTime90_ns")
    , s2RecDof(base->m_reader,"pulsesTPC.s2RecDof")
{
}

TriggerEfficiencyExtTriggerEvent::~TriggerEfficiencyExtTriggerEvent()
{
}
