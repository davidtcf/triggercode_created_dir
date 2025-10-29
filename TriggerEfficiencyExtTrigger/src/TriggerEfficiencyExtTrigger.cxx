#include "TriggerEfficiencyExtTrigger.h"
#include "Analysis.h"
#include "ConfigSvc.h"
#include "CutsBase.h"
#include "CutsTriggerEfficiencyExtTrigger.h"
#include "EventBase.h"
#include "HistSvc.h"
#include "Logger.h"
#include "SkimSvc.h"
#include "SparseSvc.h"
#include <bitset>
//Counters and things to keep track of across the full runlist
double n_rt = 0.0; //The number of events with random triggers (for monitoring the trigger efficiency)
double n_gpst = 0.0; //The number of events with gps triggers (for monitoring the trigger efficiency)
double n_tot_events = 0.0;
double n_tot_events_pre_total_area_cut = 0.0;
// Constructor
TriggerEfficiencyExtTrigger::TriggerEfficiencyExtTrigger()
    : Analysis()
    , m_evt(new TriggerEfficiencyExtTriggerEvent(m_event))
{

    m_event->Initialize();

    // Setup logging
    logging::set_program_name("TriggerEfficiencyExtTrigger Analysis");
    // Logging level: error = 1, warning = 2, info = 3, debug = 4, verbose = 5

    m_cutsTriggerEfficiencyExtTrigger = new CutsTriggerEfficiencyExtTrigger(m_evt);

    //create a config instance, this can be used to call the config variables.
    m_conf = ConfigSvc::Instance();
}

// Destructor
TriggerEfficiencyExtTrigger::~TriggerEfficiencyExtTrigger()
{
    delete m_cutsTriggerEfficiencyExtTrigger;
}

// Called before event loop
void TriggerEfficiencyExtTrigger::Initialize()
{
    INFO("Initializing TriggerEfficiencyExtTrigger Analysis");
}

// Called once per event
void TriggerEfficiencyExtTrigger::Execute()
{
    n_tot_events_pre_total_area_cut = n_tot_events_pre_total_area_cut + 1;
    
    if (m_cutsTriggerEfficiencyExtTrigger->TriggerEfficiencyExtTriggerTotalAreaCut()) {
    n_tot_events = n_tot_events + 1;
    /////////////
    //// Extract Trigger types
    /////////////
    double tpcTotArea_phd = (*m_evt->tpctotArea);
    double trgType = (*m_event->m_eventHeader)->triggerType;
    int trgTypePost = (*m_event->m_eventHeader)->triggerTypePost;
    int intTriggerType = (*m_event->m_eventHeader)->triggerType;
    bitset<8> TT = bitset<8>(intTriggerType); //Unpack TType into 8 bit binary
    bitset<8> TTP = bitset<8>(trgTypePost); //Unpack TType post into 8 bit binary
        
    //Unpack Bitsets to Individual Triggers
    bool TPCT = TT[0];//TPCTrig 
    bool SKINT = TT[1];//SkinTrig
    bool ODT = TT[2];//ODTrig
    bool GPST = TT[3];//GPSTrig
    bool EXTT = TT[4];//EXTTrig (DD Plasma Trigger)
    bool RT = TT[5];//RandomTrig
    bool MystT = TT[6];//???
    bool GCT = TT[7];//S2 Trig
    bool TPCTP = TTP[0]; //TPCTrig post
    bool SKINTP = TTP[1];//Skintrig post
    bool ODTP = TTP[2];// ODtrig post
    bool GPSTP = TTP[3];// GPStrig post
    bool EXTTP = TTP[4];// EXTtrig post
    bool RTP = TTP[5];//Randomtrig post
    bool MystTP = TTP[6];//???
    bool GCTP = TTP[7];//GlobalCoincidenceTrigPost
        
        
    
        
        
    if ((RT == true)||(GPST == true)){//((GCT == true)){////||(GPSTP == true)||(RTP == true)){//added this to just look at random triggers. Unbiased sample for monitoring
    //if (true) {
    //double rt = RT;
    //double gpst = GPST;
    //m_hists->BookFillTree("RT","RT",&rt);
    //m_hists->BookFillTree("GPST","GPST",&gpst);
    if (RT == true){
        
        n_rt = n_rt + 1;
    }
    if (GPST == true){
        n_gpst = n_gpst + 1;
    }
        
    int tcut = 1000; //number of ns pulse needs to be isolated by to count in analysis. 1000 ns = 1 us
    int window_ns = 5000; //number of ns trigger pulse can be outside the pulse boundaries to still be considered coincident. Currently 5 us
    int pulse_length_upper_bound = 50000; //maximum pulse lengths considered in ns. cuts unphysically long S2s. Currently 50 microseconds
    int pulse_length_lower_bound =   50; //minimum pulse lengths considered in ns. cuts unphysically short S2s. Currently .05 microseconds
    //Get List of pulse IDs to loop through
    
    vector<int> s1IDs = (*m_evt->s1PulseIDs);
    vector<int> s2IDs = (*m_evt->s2PulseIDs);
    vector<int> sEIDs = (*m_evt->sEPulseIDs);
    vector<int> oIDs = (*m_evt->otherPulseIDs);
    vector<int> IDs;
    
    // Concatenate the vectors
    
    if (s1IDs.size()>0){
    for(int j = 0; j<s1IDs.size();j++){
        IDs.push_back(s1IDs[j]);
    }
    }
    
    if (oIDs.size()>0){
    for(int j = 0; j<oIDs.size();j++){
        IDs.push_back(oIDs[j]);
    }
    }
        
    if (s2IDs.size() >0){
    for(int j = 0; j<s2IDs.size();j++){
        IDs.push_back(s2IDs[j]);
        
    }
    }
    
    if (sEIDs.size() > 0 ){
    for(int j = 0; j<sEIDs.size();j++){
        IDs.push_back(sEIDs[j]);
    }    
    }
    
    //IDs.push_back(5);
    if (IDs.size()>0){    
        
    sort(IDs.begin(),IDs.end());//Pulse IDs is now the list of pulse IDs we're interested in in ascending order
    
    
    for(int x=0; x < IDs.size(); x++){ //loop through each tpc pulse of interest
        int i; 
        i = IDs[x];//Get the pulse ID of interest
        int im1;
        int ip1;
        // For the first pulse in the event (x = 0) special condition
        if(i==0){
            im1 = i;
            }
        else if ( x == 0 ) {
            im1 = i - 1;
        }
        else{
            im1 = IDs[x-1];
        }
        //For the last pulse in the event
        if (i>= (*m_evt->tpcnPulses)-1){
            ip1 = i;
            }
        else if (x >= IDs.size()-1){
            ip1 = i + 1;
        }
        else{
            ip1 = IDs[x+1];
        }
        
    //for (int i = 0; i < (*m_evt->tpcnPulses); i++) {
        if (((*m_evt->pulseClassification)[i] =="SE")||((*m_evt->pulseClassification)[i] =="S2")){
        long int pulse_iminus1_EndTime_ns = (*m_evt->tpcpulseEndTime_ns)[im1];//Previous Pulse End Time
        long int pulse_i_StartTime_ns = (*m_evt->tpcpulseStartTime_ns)[i];//Pulse Start Time
        long int pulse_i_EndTime_ns = (*m_evt->tpcpulseEndTime_ns)[i];//Pulse End Time
        long int pulse_iplus1_StartTime_ns = (*m_evt->tpcpulseStartTime_ns)[ip1];//Next Pulse Start Time
        long int pulse_length_ns = pulse_i_StartTime_ns - pulse_i_EndTime_ns; //Pulse length in ns
        long int tdiff_previous = pulse_i_StartTime_ns - pulse_iminus1_EndTime_ns; //Time Separation between pulse i-1 and pulse i
        
        long int tdiff_next =pulse_iplus1_StartTime_ns - pulse_i_EndTime_ns; //Time Separation between pulse i and pulse i+1
        //m_hists->BookFillTree("Test1","tdiff_prev",&tdiff_previous);
        //m_hists->BookFillTree("Test1","tdiff_next",&tdiff_next);
        if ((tdiff_previous > tcut)&&(tdiff_next > tcut)&&(tdiff_previous > 0)&&(tdiff_next > 0 )&&(pulse_length_lower_bound<pulse_length_ns<pulse_length_upper_bound)){ //If pulse is well separated in time and less than 10 us long
            double pulseArea_phd = (*m_evt->tpcHGPulseArea)[i];//Get Pulse Area
            double tdiff_previous_dub = tdiff_previous;
            double tdiff_next_dub = tdiff_next;
            if (true){//Removed a cut that will be put back in in python
                
                double trgStatus = 0;//Start by assuming the pulse did not trigger
                //m_hists->BookFillTree("Test","tdiff_prev",&tdiff_previous);
                //m_hists->BookFillTree("Test","tdiff_next",&tdiff_next);
                double ttimebooked = 99999999;
                long int TDiff = 99999999;
                long int time_difference = 99999999;
                double nTriggerPulses = (*m_evt->trgSumnPulses);
                for (int i = 0; i < (*m_evt->trgSumnPulses); i++){//Loop Through Trigger Timestamps
                    double trgTimebooked = (*m_evt->trgSumpulseStartTime)[i]; //Start of pulse indicating trigger
                    long int trgTime = (*m_evt->trgSumpulseStartTime)[i];
                    
                    long int trgTime_PulseStartTimeDifference = abs(trgTime - pulse_i_StartTime_ns);
                    long int trgTime_PulseEndTimeDifference = abs(trgTime - pulse_i_EndTime_ns);
                    if (trgTime_PulseEndTimeDifference>trgTime_PulseStartTimeDifference){//If the time difference between the trigger time and the pulse end time is greater than the difference between the trigger time and the pulse start time
                        time_difference = trgTime_PulseStartTimeDifference;
                    }
                    else{
                        time_difference =trgTime_PulseEndTimeDifference;
                    }
                    if (time_difference < TDiff){//if the time difference is less than the previously set minimum, update the time difference and the set that pulse time as the time we save
                        TDiff = time_difference;
                        ttimebooked = trgTimebooked;
                    }
                    
                    if ((pulse_i_StartTime_ns < trgTime)&&(trgTime < pulse_i_EndTime_ns + window_ns)){ //If the time difference is less than the coincidence window we set for this analysis.
                        trgStatus = 1.;//mark the pulse as having triggered
                        
                    }
                    

                    
                    // if ((pulse_i_StartTime_ns < trgTime)&&(trgTime < pulse_i_EndTime_ns + window_ns)){//if trigger pulse is coincident with the pulse
                    
                    //     trgStatus = 1.;//Change the trigger status to true for this pulse
                    //     ttimebooked = trgTimebooked;
                    //     TDiff = 0;
                    // }
                    // else if ((*m_evt->trgSumnPulses)!=0){ //Else if it didn't trigger and there are pulses in the trigger sum, book the closest trigger time to the pulse
                    //     for (int j = 0; j <= (*m_evt->trgSumnPulses); j++){
                    //         long int trgTime = (*m_evt->trgSumpulseStartTime)[j];
                    //         long int trgTime_PulseStartTimeDifference = abs(trgTime - pulse_i_StartTime_ns);
                    //         long int trgTime_PulseEndTimeDifference = abs(trgTime - pulse_i_EndTime_ns);
                    //         if (TDiff > trgTime_PulseEndTimeDifference){
                    //             ttimebooked = (*m_evt->trgSumpulseStartTime)[j];
                    //             TDiff = trgTime_PulseEndTimeDifference;
                    //         }
                    //         if (TDiff > trgTime_PulseStartTimeDifference){
                    //             ttimebooked = (*m_evt->trgSumpulseStartTime)[j];
                    //             TDiff = trgTime_PulseStartTimeDifference;
                    //         }
                    //     }
                        
                    // }
                }
                
                vector<float> chAreas = (*m_evt->chPulseArea_phd)[i];
                
                double maxchannelarea = *max_element(chAreas.begin(), chAreas.end());
                
                //Trying something here
                sort(chAreas.begin(), chAreas.end(), std::greater<>());
                double area00 = chAreas[0];
                double area01 = chAreas[1];
                double area02 = chAreas[2];
                double area03 = chAreas[3];
                double area04 = chAreas[4];
                double area05 = chAreas[5];
                double area06 = chAreas[6];
                double area07 = chAreas[7];
                double area08 = chAreas[8];
                double area09 = chAreas[9];
                double area10 = chAreas[10];
                double area11 = chAreas[11];
                
                m_hists->BookFillTree("chAreas00","chAreas00",&area00);
                m_hists->BookFillTree("chAreas01","chAreas01",&area01);
                m_hists->BookFillTree("chAreas02","chAreas02",&area02);
                m_hists->BookFillTree("chAreas03","chAreas03",&area03);
                m_hists->BookFillTree("chAreas04","chAreas04",&area04);
                m_hists->BookFillTree("chAreas05","chAreas05",&area05);
                m_hists->BookFillTree("chAreas06","chAreas06",&area06);
                m_hists->BookFillTree("chAreas07","chAreas07",&area07);
                m_hists->BookFillTree("chAreas08","chAreas08",&area08);
                m_hists->BookFillTree("chAreas09","chAreas09",&area09);
                m_hists->BookFillTree("chAreas10","chAreas10",&area10);
                m_hists->BookFillTree("chAreas11","chAreas11",&area11);

                m_hists->BookFillTree("nTriggerPulses","nTriggerPulses",&nTriggerPulses);
                double runID = (*m_event->m_eventHeader)->runID;
                double eventID = (*m_event->m_eventHeader)->eventID;
                double pulse_i_StartTime = (*m_evt->tpcpulseStartTime_ns)[i];//Pulse Start Time
                double pulse_i_EndTime = (*m_evt->tpcpulseEndTime_ns)[i];//Pulse End Time
                
                double s2XYrecStatus =  (*m_evt->s2XYchi2)[i];
                
                double negativeArea =(*m_evt->negativeArea)[i];
                
                double pulseArea200ns_phd = (*m_evt->pulseArea200ns_phd)[i];
                
                
                
                double aft10_ns = (*m_evt->aft10)[i];
                double aft90_ns = (*m_evt->aft90)[i];
                double s2X_cm = (*m_evt->s2X_cm)[i];
                double s2Y_cm = (*m_evt->s2Y_cm)[i];
                int s2RecDofint = (*m_evt->s2RecDof)[i];
                double s2RecDofdub = s2RecDofint;
                m_hists->BookFillTree("s2RecDof","s2RecDof",&s2RecDofdub);
                m_hists->BookFillTree("aft10_ns","aft10_ns",&aft10_ns);
                m_hists->BookFillTree("aft90_ns","aft90_ns",&aft90_ns);
                m_hists->BookFillTree("s2X_cm","s2X_cm",&s2X_cm);
                m_hists->BookFillTree("s2Y_cm","s2Y_cm",&s2Y_cm);
                m_hists->BookFillTree("tpcTotArea_phd","tpcTotArea_phd",&tpcTotArea_phd);
                
                m_hists->BookFillTree("negativeArea_phd","negativeArea_phd",&negativeArea);
                m_hists->BookFillTree("pulseArea200ns_phd","pulseArea200ns_phd",&pulseArea200ns_phd);
                m_hists->BookFillTree("s2XYChi2","s2XYChi2",&s2XYrecStatus);
                
                m_hists->BookFillTree("MaxArea","MaxChArea", &maxchannelarea);
                m_hists->BookFillTree("trgType","trgType",&trgType);
                m_hists->BookFillTree("Time_Separated_all_Pulses_PST","PST",&pulse_i_StartTime);
                m_hists->BookFillTree("Time_Separated_all_Pulses_PET","PET",&pulse_i_EndTime);
                m_hists->BookFillTree("Time_Separated_all_Pulses_rid","rid",&runID);
                m_hists->BookFillTree("Time_Separated_all_Pulses_eid","eid",&eventID);
                m_hists->BookFillTree("Time_Separated_all_Pulses_ttime","ttime",&ttimebooked);
                m_hists->BookFillTree("Time_Separated_all_Pulses_trg","trgStatus",&trgStatus);
                m_hists->BookFillTree("Time_Separated_all_Pulses_Areas","Pulse_Area",&pulseArea_phd);
                m_hists->BookFillTree("Time_Separated_all_Pulses_tdiff_prev","tdiff_prev",&tdiff_previous_dub);
                m_hists->BookFillTree("Time_Separated_all_Pulses_tdiff_next","tdiff_next",&tdiff_next_dub);
                
                m_hists->BookFillTree("n_totrandomtrigs","n_totrandomtrigs",&n_rt);
                m_hists->BookFillTree("n_totgpstrigs","n_totgpstrigs",&n_gpst);
                m_hists->BookFillTree("n_totevents","n_totevents",&n_tot_events);
                m_hists->BookFillTree("n_totevents_preareacut","n_totevents_preareacut",&n_tot_events_pre_total_area_cut);
                
            }
            //m_hists->BookFillTree("Time_Separated_Pulses_trgStatus","trgStatus",trgStatus);
        }
    }
    }
/*
    // Example of using the analysis specific cuts when using individual variable loading - cutting on number of pulses
    if (m_cutsSingleVarS2Trig->SingleVarS2TrigNumPulsesGT10()) {
        VERBOSE("Passed the cut");
    }
    */
}
}
}

}

// Called after event loop
void TriggerEfficiencyExtTrigger::Finalize()
{
}
