#include "TriggerEfficiencyExtTrigger.h"
#include "ConfigSvc.h"
#include <string>

int main(int argc, char* argv[]) {


  // Crate config svc to read from unique config file
  string alpacaTopDir = std::getenv("ALPACA_TOPDIR");
  string analysisName("TriggerEfficiencyExtTrigger");
  string configFile("config/TriggerEfficiencyExtTrigger.config");
  string configFileFullPath = alpacaTopDir + "/modules/" + analysisName + "/" + configFile;

  ConfigSvc* config = ConfigSvc::Instance(argc, argv, analysisName, alpacaTopDir, configFileFullPath);

  Analysis* ana;
  // Create analysis code and run
  if(config->configFileVarMap["whichAna"]==0){
  ana = new TriggerEfficiencyExtTrigger();
  ana->Run(config->FileList, config->OutName);
  }
  // Clean up
  delete ana;
  delete config;

  return 0;
}
