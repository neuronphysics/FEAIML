#include "starmaze.h"

#include <despot/solver/pomcp.h>
#include <despot/core/builtin_lower_bounds.h>
#include <despot/core/builtin_policy.h>
#include <despot/core/builtin_upper_bounds.h>
#include <despot/core/particle_belief.h>
#include <assert.h>


using namespace std;

namespace despot {
/* =============================================================================
* SimpleState class
* =============================================================================*/
StarMazeProblem* StarMazeProblem::current_ = NULL;
SimpleState::SimpleState() {
}


SimpleState::SimpleState(int _state_id) {
	state_id = _state_id;
}

SimpleState::~SimpleState() {
}

string SimpleState::text() const {
          return "id=" + to_string(state_id);
}


/* =============================================================================
* StarMaze class
* =============================================================================*/



StarMazeProblem::StarMazeProblem() {
	current_ = this;
	Init();
}
unsigned long StarMazeProblem::histObs = 0;
/* =============================================================================
** A function to store observation
** ============================================================================*/
// Each value takes 2 bits, so a 64 bit value can hold up to 32 values.
// Precondition: x >= 0 && x <= Mask
void StarMazeProblem::store(unsigned long &n, unsigned x) {
    //store observations as bit number 
    n = (n << BitsPerValue) | x;
}

unsigned StarMazeProblem::get_max(unsigned long n){
    //retrieve the maximum value in n  
    //To mask out the lower 3 bits you would use 7 (binary 111). With 3 bits you could store values from 0 to 7.
    unsigned m = 0;
    for ( ; n; n >>= BitsPerValue) 
        if ((n & Mask) > m) 
           m = n & Mask;
    return m;
}
/*===========================================================*/
void StarMazeProblem::Init() {


	SimpleState* state;
	states_.resize(NumStates());
    obs_.resize(NumStates());
	pos_.resize(NumStates());
	cont_.resize(NumStates());
    tim_.resize(NumStates());

	for (int context = 0; context < CONTEXTTYPE; context++) {
		for (int position = 0; position < MAZEPOSITIONS; position++) {
            for (int time=0; time<TOTALTIME; time++){
			    int s = PosConTimIndicesToStateIndex(context, position, time);
			    state      = new SimpleState(s);
			    states_[s] = state;
	            pos_[s]    = position;
			    cont_[s]   = context;
                tim_[s]    = time;
                if (pos_[s]==CUE ){
                   double random_number = Random::RANDOM.NextDouble();
                   cout<<"random number:"<<random_number<<endl;
                   switch (cont_[s]) {
                        case C_LEFT:
                            obs_[s] =  (random_number > NOISE) ? O_LEFT : O_NONE; 
                            break;
                        case C_TOPLEFT:
                            obs_[s] =  (random_number > NOISE) ? O_TOPLEFT : O_NONE; 
                            break;
                        case C_RIGHT:
                            obs_[s] =  (random_number > NOISE) ? O_RIGHT : O_NONE; 
                            break;
                        case C_TOPRIGHT:
                            obs_[s] =  (random_number > NOISE) ? O_TOPRIGHT : O_NONE; 
                            break;
                    }
                    //cout<< pos_[s]<<" "<<cont_[s]<<" "<<tim_[s] <<" "<<obs_[s]<<endl;
                }else if (cont_[s]==C_RIGHT && pos_[s]==RIGHT ){
                    obs_[s] = O_RIGHT ; 
                    //cout<< pos_[s]<<" "<<cont_[s]<<" "<<tim_[s]<<" "<<obs_[s]<<endl;
                }else if(cont_[s]==C_LEFT && pos_[s]==LEFT ){
                    obs_[s] = O_LEFT;
                    //cout<< pos_[s]<<" "<<cont_[s]<<" "<<tim_[s]<<" "<<obs_[s]<<endl;
                }else if(cont_[s]==C_TOPRIGHT && pos_[s]==TOPRIGHT2 ){
                    obs_[s] =   O_TOPRIGHT ; 
                    //cout<< pos_[s]<<" "<<cont_[s]<<" "<<tim_[s]<<" "<<obs_[s]<<endl;
                }else if(cont_[s]==C_TOPLEFT && pos_[s]==TOPLEFT2 ){
                    obs_[s] =   O_TOPLEFT ; 
                    //cout<< pos_[s]<<" "<<cont_[s]<<" "<<tim_[s]<<" "<<obs_[s]<<endl;
                }else{
                    obs_[s] =   O_NONE; 
                }

            }
		}
	}

	// Build transition matrix
    //int TotalNumState=NumStates()-NumStates()/TOTALTIME; //Number of states with allowed transitions
    
	transition_probabilities_.resize(NumStates());
    //define reward table
    reward_.resize(NumStates());
    
	for (int s = 0; s < NumStates(); s++) {
        reward_[s].resize(NumActions());
		transition_probabilities_[s].resize(NumActions());
		for (int a = 0; a < NumActions(); a++) {
            transition_probabilities_[s][a].clear();
            State next;
            next.state_id = PosConTimIndicesToStateIndex(cont_[s], a, tim_[s] + 1);
            //setting reward table values
            reward_[s][a]=0.0;
            if (tim_[s]>=TIME_STEP_3){
                if ( cont_[s]==C_RIGHT ){
                    switch (pos_[s]) {
                        case CENTER: reward_[s][a]=-40.0;break;
                        case CUE: reward_[s][a]=-40.0;break;   
                        case RIGHT: reward_[s][a]=0.0;break;
                        case LEFT: reward_[s][a]=-40.0;break;
                        case TOPRIGHT1: reward_[s][a]=-40.0;break;
                        case TOPRIGHT2: reward_[s][a]=-40.0;break;
                        case TOPLEFT1: reward_[s][a]=-40.0;break;
                        case TOPLEFT2: reward_[s][a]=-40.0;break;
                    }
                    if (tim_[s]==TIME_STEP_4 && a!=A_RIGHT ){
                        reward_[s][a]=-40.0;
                    }
                }else if(cont_[s]==C_LEFT){
                    if (a==A_LEFT){
                        switch (pos_[s]) {
                            case CENTER: reward_[s][a]=10.0;break;
                            case CUE: reward_[s][a]=10.0;break;   
                            case RIGHT: reward_[s][a]=0.0;break;
                            case LEFT: reward_[s][a]=10.0;break;
                            case TOPRIGHT1: reward_[s][a]=0.0;break;
                            case TOPRIGHT2: reward_[s][a]=0.0;break;
                            case TOPLEFT1: reward_[s][a]=0.0;break;
                            case TOPLEFT2: reward_[s][a]=0.0;break;
                       }
                    }  
                }else if (tim_[s]==TIME_STEP_3 &&  cont_[s]==C_TOPRIGHT ){
                    if (a==A_TOPRIGHT2 ){
                        switch (pos_[s]) {
                            case CENTER: reward_[s][a]=-40.0;break;
                            case CUE: reward_[s][a]=-40.0;break;   
                            case RIGHT: reward_[s][a]=-40.0;break;
                            case LEFT: reward_[s][a]=-40.0;break;
                            case TOPRIGHT1: reward_[s][a]=0.0;break;
                            case TOPRIGHT2: reward_[s][a]=0.0;break;
                            case TOPLEFT1: reward_[s][a]=-40.0;break;
                            case TOPLEFT2: reward_[s][a]=-40.0;break;
                       }
                    }else{
                        reward_[s][a]=-40.0;
                    }
                }else if (tim_[s]==TIME_STEP_4 &&cont_[s]==C_TOPRIGHT ){
                    if (a!=A_TOPRIGHT2){
                        reward_[s][a]=-40.0;
                    }
                }else if(cont_[s]==C_TOPLEFT){
                    if (a==A_TOPLEFT2){
                        switch (pos_[s]) {
                            case CENTER: reward_[s][a]=0.0;break;
                            case CUE: reward_[s][a]=0.0;break;   
                            case RIGHT: reward_[s][a]=0.0;break;
                            case LEFT: reward_[s][a]=0.0;break;
                            case TOPRIGHT1: reward_[s][a]=0.0;break;
                            case TOPRIGHT2: reward_[s][a]=0.0;break;
                            case TOPLEFT1: reward_[s][a]=20.0;break;
                            case TOPLEFT2: reward_[s][a]=20.0;break;
                       }
                    }  
                }
            }
            //cout<<"position: "<<pos_[s]<<", context: "<<cont_[s]<<", time: "<<tim_[s]<<", action: "<<a<<", reward: "<<reward_[s][a]<<endl;
            if (tim_[s]<=TIME_STEP_3){
                
                if (pos_[s]==CENTER ){
                   //if the rat is at the center and if she doesn't take topleft2 or topright2 actions then the probability of transition is 0.86   
                   switch (a) {
                        case A_CENTER: next.weight = 0.965;break;
                        case A_CUE: next.weight = 0.965;break;   
                        case A_RIGHT: next.weight = 0.965;break;
                        case A_LEFT: next.weight = 0.965;break;
                        case A_TOPRIGHT1: next.weight = 0.965;break;
                        case A_TOPRIGHT2: next.weight = 0.005;break;
                        case A_TOPLEFT1: next.weight = 0.965;break;
                        case A_TOPLEFT2: next.weight = 0.005;break;
                    }
                }else if ( pos_[s]==CUE){
                    
                    switch (a) {
                        case A_CENTER: next.weight = 0.005;break;
                        case A_CUE: next.weight = 0.965;break;   
                        case A_RIGHT: next.weight = 0.965;break;
                        case A_LEFT: next.weight = 0.965;break;
                        case A_TOPRIGHT1: next.weight = 0.965;break;
                        case A_TOPRIGHT2: next.weight = 0.005;break;
                        case A_TOPLEFT1: next.weight = 0.965;break;
                        case A_TOPLEFT2: next.weight = 0.005;break; 
                    }
                }else if(pos_[s]==RIGHT){
                    
                    switch (a) {
                        //if the rat is at the right arm then the most likely transition will be to stay (absorbing state)
                        case A_CENTER: next.weight = 0.01;break;
                        case A_CUE: next.weight = 0.01;break;   
                        case A_RIGHT: next.weight = 0.93;break;
                        case A_LEFT: next.weight = 0.01;break;
                        case A_TOPRIGHT1: next.weight = 0.01;break;
                        case A_TOPRIGHT2: next.weight = 0.01;break;
                        case A_TOPLEFT1: next.weight = 0.01;break;
                        case A_TOPLEFT2: next.weight = 0.01;break; 
                    }  
                }else if(pos_[s]==LEFT){
                    //if the rat is at the left arm then the most likely transition will be to stay
                    switch (a) {
                        case A_CENTER: next.weight = 0.01;break;
                        case A_CUE: next.weight = 0.01;break;    
                        case A_RIGHT: next.weight = 0.01;break;
                        case A_LEFT: next.weight = 0.93;break;
                        case A_TOPRIGHT1: next.weight = 0.01;break;
                        case A_TOPRIGHT2: next.weight = 0.01;break;
                        case A_TOPLEFT1: next.weight = 0.01;break;
                        case A_TOPLEFT2: next.weight = 0.01;break; 
                    }  
                }else if(pos_[s]==TOPRIGHT1){
                    //if the rat is at the topright1, the only likely transition is to go to the topright2  
                    switch (a) {
                        case A_CENTER: next.weight = 0.005;break;
                        case A_CUE:next.weight = 0.005;break;    
                        case A_RIGHT: next.weight = 0.005;break;
                        case A_LEFT: next.weight = 0.005;break;
                        case A_TOPRIGHT1: next.weight = 0.005;break;
                        case A_TOPRIGHT2: next.weight = 0.965;break;
                        case A_TOPLEFT1: next.weight = 0.005;break;
                        case A_TOPLEFT2: next.weight = 0.005;break; 
                    }  
                }else if(pos_[s]==TOPRIGHT2){
                    //if the rat is at the topright2, the only likely transition is to stay put 
                    switch (a) {
                        case A_CENTER: next.weight = 0.005;break;
                        case A_CUE: next.weight = 0.005;break; 
                        case A_RIGHT: next.weight = 0.005;break;
                        case A_LEFT: next.weight = 0.005;break;
                        case A_TOPRIGHT1: next.weight = 0.005;break;
                        case A_TOPRIGHT2: next.weight = 0.965;break;
                        case A_TOPLEFT1: next.weight = 0.005;break;
                        case A_TOPLEFT2: next.weight = 0.005;break; 
                    }     
                }else if(pos_[s]==TOPLEFT1){
                    //if the rat is at the topleft1 arm then the most likely to go to topleft2
                    switch (a) {
                        case A_CENTER: next.weight = 0.005;break;
                        case A_CUE:next.weight = 0.005;break; 
                        case A_RIGHT: next.weight = 0.005;break;
                        case A_LEFT: next.weight = 0.005;break;
                        case A_TOPRIGHT1: next.weight = 0.005;break;
                        case A_TOPRIGHT2: next.weight = 0.005;break;
                        case A_TOPLEFT1: next.weight = 0.005;break;
                        case A_TOPLEFT2: next.weight = 0.965;break; 
                    }     
                }else if(pos_[s]==TOPLEFT2){
                    //if the rat is at the toplef2t arm then the most likely transition will be to stay
                    switch (a) {
                        case A_CENTER: next.weight = 0.005;break;
                        case A_CUE: next.weight = 0.005;break;    
                        case A_RIGHT: next.weight = 0.005;break;
                        case A_LEFT: next.weight = 0.005;break;
                        case A_TOPRIGHT1: next.weight = 0.005;break;
                        case A_TOPRIGHT2: next.weight = 0.005;break;
                        case A_TOPLEFT1: next.weight = 0.005;break;
                        case A_TOPLEFT2: next.weight = 0.965;break; 
                    }    
                }
            }else{
                //transitions with zero probabilities
                int total = TOTALTIME*MAZEPOSITIONS ;
                if (next.state_id%total==0){
                    next.state_id -= total;
                }
                switch (a) {
                    case A_CENTER: next.weight = 0.0;break;
                    case A_CUE: next.weight = 0.0;break;   
                    case A_RIGHT: next.weight = 0.0;break;
                    case A_LEFT: next.weight = 0.0;break;
                    case A_TOPRIGHT1: next.weight = 0.0;break;
                    case A_TOPRIGHT2: next.weight = 0.0;break;
                    case A_TOPLEFT1: next.weight = 0.0;break;
                    case A_TOPLEFT2: next.weight = 0.0;break;
                }
                
            }
           transition_probabilities_[s][a].push_back(next);
		}
	}
    //PrintTransitions();
}

StarMazeProblem::~StarMazeProblem() {

}
/* =============================================================================
* OptimalStarMazePolicy class
* =============================================================================*/

class OptimalStarMazePolicy: public DefaultPolicy {
private:
        const StarMazeProblem* Starmaze_;
public:
        
        OptimalStarMazePolicy(const DSPOMDP* model,
               ParticleLowerBound* bound) :
               DefaultPolicy(model, bound), Starmaze_(static_cast<const StarMazeProblem*>(model)) {
        }

        // NOTE: optimal action
        ACT_TYPE Action(const vector<State*>& particles, RandomStreams& streams,
            History& history) const {

            //similar to the tiger problem
            int count_diff = 0;
		    for (int i = history.Size() - 1; i >= 0 && history.Action(i) ==StarMazeProblem::A_CUE; i--){
                if (history.Observation(i) == StarMazeProblem::O_LEFT){
                    count_diff +=1;
                }else if (history.Observation(i) == StarMazeProblem::O_TOPLEFT){
                    count_diff +=3;
                }else if (history.Observation(i) == StarMazeProblem::O_RIGHT){
                    count_diff -=1;
                }else if(history.Observation(i) == StarMazeProblem::O_TOPRIGHT){
                    count_diff -=3;
                }
            }    
			
		    if (count_diff >= 2 &&  count_diff < 4)
			   return StarMazeProblem::A_LEFT;
            else if (count_diff >= 6)
			   return StarMazeProblem::A_TOPLEFT1;
		    else if (count_diff > -4 && count_diff <= -2)
			   return StarMazeProblem::A_RIGHT;
            else if (count_diff <= -6)
			   return StarMazeProblem::A_TOPRIGHT1;
		    else
			   return StarMazeProblem::A_CUE;
        }
};


/* ==============================
 * Deterministic simulative model
 * ==============================*/

bool StarMazeProblem::Step(State& s, double rand_num, ACT_TYPE action,
        double& reward, OBS_TYPE& obs) const {
    SimpleState& state = static_cast < SimpleState& >(s);
    
    bool terminal = false;
    reward=reward_[state.state_id][action];
    if (tim_[state.state_id]==TIME_STEP_4){//exit condition
        
        terminal = true;
        return terminal;
    }
              
    
    const vector<State>& distribution =
		   transition_probabilities_[state.state_id][action];
	double sum = 0;
	for (int i = 0; i < distribution.size(); i++) {
	    const State& next = distribution[i];
		sum += next.weight;
		if (sum >= rand_num) {
			state.state_id = next.state_id;
            //add new observation to the history
            break;
		}
	}
    //retrive the maximum value stored in the history 
    obs = obs_[state.state_id];
    return terminal;
     

}

/*=======================================
 *
 *=======================================*/


int StarMazeProblem::NumStates() const {
	 return CONTEXTTYPE * MAZEPOSITIONS * TOTALTIME;
}


double StarMazeProblem::ObsProb(OBS_TYPE obs, const State& state,
    ACT_TYPE action) const {
      return obs==obs_[state.state_id];
}
void StarMazeProblem::PrintMDPPolicy() const {
	cout << "MDP (Start)" << endl;
	for (int s = 0; s < NumStates(); s++) {
		cout << "State " << s << "; Action = " << policy_[s].action
			<< "; Reward = " << policy_[s].value << endl;
		PrintState(*(states_[s]));
	}
	cout << "MDP (End)" << endl;
}
void StarMazeProblem::PrintTransitions() const {
	cout << "Transitions (Start)" << endl;
	for (int s = 0; s < NumStates(); s++) {
		cout
			<< "--------------------------------------------------------------------------------"
			<< endl;
		cout << "State " << s << endl;
		PrintState(*GetState(s));
		for (int a = 0; a < NumActions(); a++) {
			cout << transition_probabilities_[s][a].size()
				<< " outcomes for action " << a << endl;
			for (int i = 0; i < transition_probabilities_[s][a].size(); i++) {
				const State& next = transition_probabilities_[s][a][i];
				cout << "Next = (" << next.state_id << ", " << next.weight
					<< ")" << endl;
				PrintState(*GetState(next.state_id));
			}
		}
	}
	cout << "Transitions (End)" << endl;
}

/* ================================================
* Functions related to beliefs and starting states
* ================================================*/

State* StarMazeProblem::CreateStartState(string type) const {
  // Always rat starts at the center at time 0 with uniform belief about the context
   
   int context= rand()%CONTEXTTYPE;
   int s = PosConTimIndicesToStateIndex(context,CENTER,TIME_STEP_1);
   return new SimpleState(s);//????
}

Belief* StarMazeProblem::InitialBelief(const State* start, string type) const {
        
        if (type == "DEFAULT" || type == "PARTICLE") {
            vector<State*> particles;
            for (int cont = 0; cont!=CONTEXTTYPE; ++cont) {
                
                int s = PosConTimIndicesToStateIndex( cont, CENTER, TIME_STEP_1);
                //Allocate() function allocates some space for creating new state;
                SimpleState* InitialState = static_cast<SimpleState*>(Allocate(s, 0.25));
                particles.push_back(InitialState);
            }
            return new ParticleBelief(particles, this);
        } else {
            cerr << "Unsupported belief type: " << type << endl;
            exit(1);
        }
}
/* ========================
* Bound-related functions.
* ========================*/
double StarMazeProblem::Reward(int s, ACT_TYPE action) const {
    const SimpleState* simple_state = states_[s];
    double reward=reward=reward_[simple_state->state_id][action];
	
	return reward;
}


double StarMazeProblem::GetMaxReward() const {
       return 20;
}

ValuedAction StarMazeProblem::GetBestAction() const {
		return ValuedAction(A_CUE, 0);
}
/*problematic parts*/
void StarMazeProblem::ComputeDefaultActions(string type) const {
	cerr << "Default action = " << type << endl;
	if (type == "MDP") {
		const_cast<StarMazeProblem*>(this)->ComputeOptimalPolicyUsingVI();
		int num_states = NumStates();
		default_action_.resize(num_states);

		double value = 0;//it seems this variable is redundant 
		for (int s = 0; s < num_states; s++) {
			default_action_[s] = policy_[s].action;
			value += policy_[s].value;
		}
	} else {
		cerr << "Unsupported default action type " << type << endl;
		exit(0);
	}
}

//????
const vector<State>& StarMazeProblem::TransitionProbability(int s, ACT_TYPE a) const {
	return transition_probabilities_[s][a];
}
//????what is this function doing?? 
Belief* StarMazeProblem::Tau(const Belief* belief, ACT_TYPE action,
	OBS_TYPE obs) const {
       
	static vector<double> probs = vector<double>(NumStates());

	const vector<State*>& particles =
		static_cast<const ParticleBelief*>(belief)->particles();
  //********************

	double sum = 0;
	for (int i = 0; i < particles.size(); i++) {
		
		SimpleState* state = static_cast<SimpleState*>(particles[i]);

        const vector<State>& distribution = transition_probabilities_[GetIndex(
			state)][action];

        for (int j = 0; j < distribution.size(); j++) {
		    const State& next = distribution[j];

		    double p = state->weight * next.weight*ObsProb(obs, next, action);
		    probs[next.state_id] += p;
		    sum += p;
        }
	}
  //******************
	vector<State*> new_particles;
	for (int i = 0; i < NumStates(); i++) {
		if (probs[i] > 0) {
			State* new_particle = Copy(states_[i]);
			new_particle->weight = probs[i] / sum;
			new_particles.push_back(new_particle);
			probs[i] = 0;
		}
	}

	if (new_particles.size() == 0) {
		cout << *belief << endl;
		exit(0);
	}

	return new ParticleBelief(new_particles, this, NULL, false);
}


/*=====================================*
 *     Bound
 *=====================================*/

ParticleUpperBound* StarMazeProblem::CreateParticleUpperBound(string name) const {
        if (name == "TRIVIAL" ) {
            return new TrivialParticleUpperBound(this);
        } else if (name == "MDP"|| name == "DEFAULT") {
            return new MDPUpperBound(this, *this);
        } else {
            cerr << "Unsupported particle lower bound: " << name << endl;
            exit(1);
            return NULL;
        }
    }
ScenarioUpperBound* StarMazeProblem::CreateScenarioUpperBound(string name,
        string particle_bound_name) const {
        
        const StateIndexer* indexer = this;
        if (name == "TRIVIAL" || name == "DEFAULT") {
            return new TrivialParticleUpperBound(this);
        } else if (name == "LOOKAHEAD") {
            return new LookaheadUpperBound(this, *this,
                 CreateParticleUpperBound(particle_bound_name));

        } else if (name == "MDP") {
            return new MDPUpperBound(this, *indexer);
        } else {
            cerr << "Unsupported base upper bound: " << name << endl;
            exit(0);
        }
        
}
ScenarioLowerBound* StarMazeProblem::CreateScenarioLowerBound(string name,
                                     string particle_bound_name="DEFAULT") const {
        const DSPOMDP* model = this;
	    const StateIndexer* indexer = this;
	    const StatePolicy* policy = this;                                 
        ScenarioLowerBound* bound = NULL;
        
        if (name == "TRIVIAL" ) {
            bound = new TrivialParticleLowerBound(this);
        } else if (name == "RANDOM") {
            bound = new RandomPolicy(this,
                          CreateParticleLowerBound(particle_bound_name));
        } else if (name == "MODE" || name == "DEFAULT") {
		    ComputeDefaultActions("MDP");
		     bound = new ModeStatePolicy(model, *indexer, *policy,
			              CreateParticleLowerBound(particle_bound_name));                             
        } else if (name == "CENTER") {
            bound = new BlindPolicy(this, CENTER,
                                    CreateParticleLowerBound(particle_bound_name));
        } else if (name == "CUE") {
            bound = new BlindPolicy(this, CUE,
                                    CreateParticleLowerBound(particle_bound_name));
        } else if (name == "RIGHT") {
            bound = new BlindPolicy(this, RIGHT,
                                    CreateParticleLowerBound(particle_bound_name));
        } else if (name == "LEFT") {
            bound = new BlindPolicy(this, LEFT,
                                    CreateParticleLowerBound(particle_bound_name));
        } else if (name == "TOPRIGHT1") {
            bound = new BlindPolicy(this, TOPRIGHT1,
                                    CreateParticleLowerBound(particle_bound_name));
        } else if (name == "TOPRIGHT2") {
            bound = new BlindPolicy(this, TOPRIGHT2,
                                    CreateParticleLowerBound(particle_bound_name));
        } else if (name == "TOPLEFT1") {
            bound = new BlindPolicy(this, TOPLEFT1,
                                    CreateParticleLowerBound(particle_bound_name));
        } else if (name == "TOPLEFT2") {
            bound = new BlindPolicy(this, TOPLEFT2,
                                    CreateParticleLowerBound(particle_bound_name));
        } else if (name == "OPTIMAL") {
            bound = new OptimalStarMazePolicy(this,
                                              CreateParticleLowerBound(particle_bound_name));
        } else {
            cerr << "Unsupported scenario lower bound: " << name << endl;
            exit(1);
        }
        return bound;
}
/*end of uncertain parts*/
/* ============================================
 *  print different elemens of the POMDP model
 * ============================================*/
void StarMazeProblem::PrintState(const State& state, ostream& out) const {
        const SimpleState& simple_state = static_cast<const SimpleState&>(state);
        int s= simple_state.state_id;

        out << "Rat = " << pos_[s] << "; Context = "
        << cont_[s] << "; Time step = " << tim_[s] << endl;
}

void StarMazeProblem::PrintBelief(const Belief& belief, ostream& out) const {
    /*vector<State*> particles = static_cast<const ParticleBelief&>(belief).particles();
	for (int i = 0; i < particles.size(); i++) {
		const SimpleState* simple_state = static_cast<const SimpleState*>(particles[i]);
    
	
	}*/ //Todo
}

void StarMazeProblem::PrintObs(const State& state, OBS_TYPE obs, ostream& out) const {
        switch (obs) {
            case O_NONE:
                out << "None " << endl;
                 
                break;
            case O_LEFT:
                out << "small reward at left arm "  << endl;
                break;
            case O_TOPLEFT:
                out << "larg reward at the end of top-left arm "  << endl;
                break;
            case O_RIGHT:
                out << "Shock if at the last time step rat doesn't reach right arm " << endl;
                break;
            case O_TOPRIGHT:
                out << "Shock if at the last time step rat doesn't reach to the end of top-right arm "  << endl;
                break;
        }
}


void StarMazeProblem::PrintAction(ACT_TYPE action, ostream& out) const {
        if (action == A_LEFT) {
            cout << "Move left arm" << endl;
        } else if (action == A_RIGHT) {
            cout << "Move right arm" << endl;
        } else if (action == A_TOPLEFT1) {
            cout << "Move top-left 1 position" << endl;
        } else if (action == A_TOPRIGHT1) {
            cout << "Move top right 1 position" << endl;
        } else if (action == A_TOPLEFT2) {
            cout << "Move top-left 2 position" << endl;
        } else if (action == A_TOPRIGHT2) {
            cout << "Move top right 2 position" << endl;
        } else if (action == A_CENTER){
            cout<< "Move to center" << endl;
        }else {
            cout << "check the cue" << endl;
        }
}

/* =================
 * Memory management
 * =================*/

State* StarMazeProblem::Allocate(int state_id, double weight) const {
        
        SimpleState* state = memory_pool_.Allocate();
        state->state_id = state_id;
        state->weight = weight;
        return state;
}

State* StarMazeProblem::Copy(const State* particle) const {
        SimpleState* state = memory_pool_.Allocate();
        *state = *static_cast<const SimpleState*>(particle);
        state->SetAllocated();
        return state;
}
void StarMazeProblem::Free(State* particle) const {
        memory_pool_.Free(static_cast<SimpleState*>(particle));
}
int StarMazeProblem::NumActiveParticles() const {
	return memory_pool_.num_allocated();
}

int StarMazeProblem::GetAction(const State& state) const {
	return default_action_[GetIndex(&state)];
}

}// namespace despot
