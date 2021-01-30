#ifndef STARMAZEPROBLEM_H
#define STARMAZEPROBLEM_H
#include <despot/core/builtin_lower_bounds.h>
#include <despot/core/builtin_policy.h>
#include <despot/core/builtin_upper_bounds.h>
#include <despot/core/particle_belief.h>
#include <despot/interface/pomdp.h>
#include <despot/core/mdp.h>
using namespace std;
namespace despot {

/* =============================================================================
 * SimpleState class
 * =============================================================================*/

class SimpleState: public State {
public:
        SimpleState();
        SimpleState(int _state_id);
        ~SimpleState();

        std::string text() const;
};


/* =============================================================================
 * StarMazeProblem class
 * =============================================================================*/

class StarMazeProblem : public DSPOMDP,
     public MDP,
     public StateIndexer,
     public StatePolicy  {
     friend class SimpleState;
     friend class OptimalStarMazePolicy;

public:
        enum {//action
           A_CENTER = 0, A_CUE = 1, A_RIGHT = 2, A_LEFT = 3, A_TOPRIGHT1 = 4, A_TOPRIGHT2 = 5, A_TOPLEFT1 = 6, A_TOPLEFT2 = 7
        };
        enum { // observation
           O_NONE=0, O_LEFT=1, O_TOPLEFT=2, O_RIGHT=3, O_TOPRIGHT = 4
        };
        enum { // context
           C_LEFT=0, C_TOPLEFT=1, C_RIGHT=2, C_TOPRIGHT = 3
        };
        enum { // rat position
           CENTER = 0, CUE = 1, RIGHT = 2, LEFT = 3, TOPRIGHT1 = 4, TOPRIGHT2 = 5, TOPLEFT1 = 6, TOPLEFT2 = 7
        };
        enum { // time
           TIME_STEP_1=0, TIME_STEP_2=1, TIME_STEP_3=2, TIME_STEP_4 = 3
        };
        const int  CONTEXTTYPE = 4, MAZEPOSITIONS = 8, TOTALTIME = 4; 
        const double NOISE=0.05;

protected:
        std::vector<std::vector<std::vector<State> > > transition_probabilities_; //state, action, [state, weight]

        mutable MemoryPool<SimpleState> memory_pool_;

        std::vector<SimpleState*> states_;
        std::vector<OBS_TYPE> obs_;
   
        mutable std::vector<ValuedAction> mdp_policy_;
        mutable std::vector<ACT_TYPE> default_action_;
        std::vector<int> pos_; // pos_[s]: position of rat for state s	     
        std::vector<int> cont_; // cont_[s]: context of maze
        std::vector<int> tim_; //tim_[s]: time 
        std::vector<std::vector<double>> reward_;
private:
      // a few extra function I added to track observations in a trial
      static unsigned long histObs;
      static const unsigned BitsPerValue = 3;
      static const unsigned Mask = (1u << BitsPerValue) - 1;

public:
      StarMazeProblem();
      static StarMazeProblem* current_;
      static void store(unsigned long &n, unsigned x);
      static unsigned get_max(unsigned long n);
      void Init();
      ~StarMazeProblem(); 
      inline int PosConTimIndicesToStateIndex(int context, int rat_position, int time_step) const {                                                                                                                                                                                                                                                   
         return context*TOTALTIME*MAZEPOSITIONS
                +rat_position*TOTALTIME
                +time_step;
      }
      inline int StateIndexToContIndex(int index) const {
		   return index / (TOTALTIME*MAZEPOSITIONS);
	   }
	   inline int StateIndexToPosIndex(int index) const {
		   return index % (TOTALTIME*MAZEPOSITIONS)/TOTALTIME;
	   }
      inline int StateIndexToTimIndex(int index) const {
		   return index % TOTALTIME;
	   }

      /* Deterministic simulative model.*/
      //simulative model
      bool Step(State& state, double rand_num, ACT_TYPE action, double& reward,
              OBS_TYPE& obs) const;

      int NumStates() const;
      /* Returns total number of actions.*/
      inline int NumActions() const{
         return MAZEPOSITIONS;
      }
      /* Functions related to beliefs and starting states.*/
      
      double ObsProb(OBS_TYPE obs, const State& state, ACT_TYPE action) const;

      void PrintMDPPolicy() const; 
      void PrintTransitions() const;
      State* CreateStartState(std::string type = "DEFAULT") const;
      Belief* InitialBelief(const State* start, std::string type = "DEFAULT") const;
      double Reward(int s, ACT_TYPE a) const;
      
      /* Bound-related functions.*/
      double GetMaxReward() const;
      ValuedAction GetBestAction() const;

      void ComputeDefaultActions(std::string type)  const;

      const std::vector<State>& TransitionProbability(int s, ACT_TYPE a) const;//?
      Belief* Tau(const Belief* belief, ACT_TYPE action, OBS_TYPE obs) const;//?

      ParticleUpperBound* CreateParticleUpperBound(std::string name = "DEFAULT") const;//?
      ScenarioUpperBound* CreateScenarioUpperBound(std::string name = "DEFAULT",
              std::string particle_bound_name = "DEFAULT") const;

      ScenarioLowerBound* CreateScenarioLowerBound(std::string name ,
              std::string particle_bound_name ) const;
      /* Display.*/
      void PrintState(const State& state, std::ostream& out = std::cout) const;
      void PrintBelief(const Belief& belief, std::ostream& out = std::cout) const;
      //print observation
      void PrintObs(const State& state, OBS_TYPE observation,
          std::ostream& out = std::cout) const;
      void PrintAction(ACT_TYPE action, std::ostream& out = std::cout) const;

      /* Memory management.*/
      State* Allocate(int state_id, double weight) const;
      State* Copy(const State* particle) const;
      void Free(State* particle) const;
      int NumActiveParticles() const;

      inline int GetIndex(const State* state) const {
		return state->state_id;
      }//?
      inline const State* GetState(int index) const {
		return states_[index];
      }
      int GetAction(const State& state) const;
      

};
/* =========================================
   StarMazeBelief Class
 * =========================================*/

class StarMazeBelief: public ParticleBelief {
private:
 	     const StarMazeProblem* Starmaze_;
public:
      	StarMazeBelief(std::vector<State*> particles, const DSPOMDP* model, Belief* prior =	NULL);
      	void Update(ACT_TYPE action, OBS_TYPE obs);
};


};  // namespace despot
#endif