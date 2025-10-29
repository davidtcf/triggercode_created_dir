#ifndef TriggerEfficiencyExtTriggerEVENT_H
#define TriggerEfficiencyExtTriggerEVENT_H

#include "EventBase.h"

#include <string>
#include <vector>

#include "RQs.h"

class TriggerEfficiencyExtTriggerEvent : public EventBase {

public:
    TriggerEfficiencyExtTriggerEvent(EventBase* eventBase);
    virtual ~TriggerEfficiencyExtTriggerEvent();

    //  GET STRUCTURE AND VARIABLE TYPE FROM modules/rqlib/
    TTreeReaderValue<int> tpcnPulses;
    TTreeReaderValue<vector<float>> tpcHGPulseArea;
    TTreeReaderValue<vector<int>> tpcpulseStartTime_ns;
    TTreeReaderValue<vector<int>> tpcpulseEndTime_ns;
    TTreeReaderValue<int> trgSumnPulses;
    TTreeReaderValue<vector<int>> trgSumpulseStartTime;
    TTreeReaderValue<int> trgSumCHID;
    TTreeReaderValue<float> tpctotArea;
    TTreeReaderValue<vector<string>> pulseClassification;
    TTreeReaderValue<vector<vector<float>>> chPulseArea_phd;
    TTreeReaderValue<vector<int>> s1PulseIDs;
    TTreeReaderValue<vector<int>> s2PulseIDs;
    TTreeReaderValue<vector<int>> sEPulseIDs;
    TTreeReaderValue<vector<int>> otherPulseIDs;
    TTreeReaderValue<vector<float>> negativeArea;
    TTreeReaderValue<vector<float>> pulseArea200ns_phd;
    TTreeReaderValue<vector<float>> s2XYchi2;
    TTreeReaderValue<vector<float>> s2X_cm;
    TTreeReaderValue<vector<float>> s2Y_cm;
    TTreeReaderValue<vector<int>> aft10;
    TTreeReaderValue<vector<int>> aft90;
    TTreeReaderValue<vector<int>> s2RecDof;

private:
};

#endif // TriggerEfficiencyExtTriggerEVENT_H
