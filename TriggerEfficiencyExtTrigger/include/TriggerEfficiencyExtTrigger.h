#ifndef TriggerEfficiencyExtTrigger_H
#define TriggerEfficiencyExtTrigger_H

#include "Analysis.h"

#include "CutsTriggerEfficiencyExtTrigger.h"
#include "EventBase.h"

#include <TTreeReader.h>
#include <string>

#include "TriggerEfficiencyExtTriggerEvent.h"

class SkimSvc;

class TriggerEfficiencyExtTrigger : public Analysis {

public:
    TriggerEfficiencyExtTrigger();
    ~TriggerEfficiencyExtTrigger();

    void Initialize();
    void Execute();
    void Finalize();

protected:
    CutsTriggerEfficiencyExtTrigger* m_cutsTriggerEfficiencyExtTrigger;
    ConfigSvc* m_conf;
    TriggerEfficiencyExtTriggerEvent* m_evt;
};

#endif
