#include "CutsTriggerEfficiencyExtTrigger.h"
#include "ConfigSvc.h"

CutsTriggerEfficiencyExtTrigger::CutsTriggerEfficiencyExtTrigger(TriggerEfficiencyExtTriggerEvent* TriggerEfficiencyExtTrigger_Event)
{
    m_evt = TriggerEfficiencyExtTrigger_Event;
}

CutsTriggerEfficiencyExtTrigger::~CutsTriggerEfficiencyExtTrigger()
{
}

// Function that lists all of the common cuts for this Analysis
bool CutsTriggerEfficiencyExtTrigger::TriggerEfficiencyExtTriggerCutsOK()
{
    // List of common cuts for this analysis into one cut
    return true;
}
bool CutsTriggerEfficiencyExtTrigger::TriggerEfficiencyExtTriggerNumPulsesGT10()
{

    if ((*m_evt->tpcnPulses) > 10) {
        std::cout << "(*m_evt->tpcnPulses)  = " << (*m_evt->tpcnPulses) << ", returning true " << std::endl;
        return true;
    } else {
        std::cout << "(*m_evt->tpcnPulses)  = " << (*m_evt->tpcnPulses) << ", returning false " << std::endl;
        return false;
    }
}
//Added Cuts below
bool CutsTriggerEfficiencyExtTrigger::TriggerEfficiencyExtTriggerTotalAreaCut(){
    bool pass_cut = false; //Starts not passing
    if ((*m_evt->tpctotArea)<90000){
        pass_cut = true;
    }
    return pass_cut;
}
