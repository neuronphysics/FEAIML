#include <despot/planner.h>

#include "starmaze.h"

using namespace despot;

class MyPlanner: public Planner {
public:
  MyPlanner() {
  }

    DSPOMDP* InitializeModel(option::Option* options) {
    DSPOMDP* model = new StarMazeProblem();
    return model;
  }

  World* InitializeWorld(std::string&  world_type, DSPOMDP* model, option::Option* options)
  {
      
      return InitializePOMDPWorld(world_type, model, options);
  }

  void InitializeDefaultParameters() {
  }

  std::string ChooseSolver(){
	  return "DESPOT";
  }
};

int main(int argc, char* argv[]) {
  srand(time(NULL));
  return MyPlanner().RunEvaluation(argc, argv);
}
