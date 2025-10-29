#ifndef CutsTriggerEfficiencyExtTrigger_H
#define CutsTriggerEfficiencyExtTrigger_H

#include "EventBase.h"
#include "TriggerEfficiencyExtTriggerEvent.h"

class CutsTriggerEfficiencyExtTrigger {

public:
    CutsTriggerEfficiencyExtTrigger(TriggerEfficiencyExtTriggerEvent* m_evt);
    ~CutsTriggerEfficiencyExtTrigger();
    bool TriggerEfficiencyExtTriggerCutsOK();
    bool TriggerEfficiencyExtTriggerTotalAreaCut();
    bool TriggerEfficiencyExtTriggerNumPulsesGT10();
    
private:
    TriggerEfficiencyExtTriggerEvent* m_evt;
};

#endif
