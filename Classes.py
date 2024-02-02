import json #import for json I/O operations
from tqdm import tqdm #progress bar
from itertools import product, permutations, combinations #import for efficient implementation of combinatorics functions
from random import sample, choice #for subsampling from large clause sets
from math import exp

#unit tests class definition
class Unit_Tests(object):
    """
    implements methods for unit testing various functions
    """

    @staticmethod
    def pause_and_exit_on_key_press():
        """
        allows for program debugging by waiting for user input and keyboard interrupt,
        with exiting after input
        """
        try:
            input("Press any key to exit ..")
            exit()
        except:
            exit()

    @staticmethod
    def pause_and_continue_on_key_press():
        """
        allows for program debugging by pausing for user input,
        without exiting after input
        """
        try:
            input("Press any key to continue")
        except:
            print ("exception occured, exiting program ... ")
            exit()

    @staticmethod
    def get_db():
        """
        returns a toy database
        """
        db = list()
        db.append('next(b,c)')
        db.append('next(c,d)')
        db.append('next(a,b)')
        db.append('next(h,i)')
        db.append('next(g,h)')
        db.append('next(e,f)')
        db.append('move_left(a)')
        db.append('move_left(e)')
        db.append('move_left(g)')
        db.append('move_right(d)')
        db.append('move_right(f)')
        db.append('move_right(i)')

        return db
    
    @staticmethod
    def get_test_clause():
        """
        returns a test clause corresponding to the toy db
        """
        return 'move_left(V0) :- move_right(V1); next(V0, V1)'
    
    @staticmethod
    def get_test_target():
        """
        returns a test target corresponding to the toy db
        """
        return 'move_left/1'

#synthetic data class definition
class Synth_Data(object):
    """
    implements methods for synthetic stock trading scenario

    RULES OF THE GAME:
    consecutive buys or sells result in losses
    alternating buys and sells results in profits
    """

    def __init__(self,dataset="stock_trading"):
        """
        class constructor, creates databases,
        and adds facts, positive, and negative targets
        """
        self.facts = list()
        self.positive_targets = list()
        self.negative_targets = list()

        #facts
        self.facts.append('buy(a,t0)')
        self.facts.append('sell(a,t1)')
        self.facts.append('sell(a,t2)')
        self.facts.append('buy(b,t0)')
        self.facts.append('sell(b,t1)')
        self.facts.append('buy(b,t2)')
        self.facts.append('buy(c,t0)')
        self.facts.append('buy(c,t1)')
        self.facts.append('next(t0,t1)')
        self.facts.append('next(t1,t2)')

        #positive targets
        self.positive_targets.append('profit(a,t1)')
        self.positive_targets.append('profit(b,t1)')
        self.positive_targets.append('profit(b,t2)')

        #negative targets
        self.negative_targets.append('profit(c,t1)')
        self.negative_targets.append('profit(a,t2)')

    def get_action_set(self):
        """
        returns valid actions
        """
        return {'buy(a)': ['a'],  #can either buy or sell stocks 'a', 'b', or 'c'
                'buy(b)': ['b'], 
                'buy(c)': ['c'], 
                'sell(a)': ['a'], 
                'sell(b)': ['b'], 
                'sell(c)': ['c']}

    def get_test_clause(self):
        """
        returns a clause for testing functionality
        """
        return 'profit(V0,V2) :- sell(V0,V2); buy(V0,V1); next(V1,V2)'

    def get_db(self):
        """
        returns synthetic database
        """
        return self.facts+self.positive_targets+self.negative_targets
    
    def get_test_facts(self):
        """
        return facts to test inference
        """
        times = []
        for fact in self.facts:
            times.append(int(fact.split(',')[-1][1:][:-1]))
        return self.facts, max(times)
    
    def get_target(self):
        """
        returns target corresponding to synthetic database
        """
        return 'profit/2'
    
    def get_ideal_candidate_clauses(self):
        """
        returns ideal candidate clauses for testing
        """
        return ['profit(V0,V2) :- sell(V0,V2); buy(V0,V1); next(V1,V2)',
                'profit(V0,V2) :- buy(V0,V2); sell(V0,V1); next(V1,V2)',
                'profit(V0,V2) :- buy(V0,V2); buy(V0,V1); next(V1,V2)',
                'profit(V0,V2) :- sell(V0,V2); sell(V0,V1); next(V1,V2)']
    
    def get_knowledge_clauses(self):
        """
        returns ideal knowledge clauses for this synthetic setting,
        This should be read from the trainer_config file
        """
        return ['profit(V0,V2) :- sell(V0,V2); buy(V0,V1); next(V1,V2)',
                'profit(V0,V2) :- buy(V0,V2); sell(V0,V1); next(V1,V2)']


    def __repr__(self):
        """
        returns string representation for objects of this class
        """
        return str(self.__dict__)

#clause prover class definition    
class Prover(object):
    """
    implements method for predicate clause proving
    """
    config = None

    @staticmethod
    def set_config():
        """
        sets configuration from prover config.json file, 
        e.g., max clause length
        """ 
        config_file_pointer = open('prover_config.json') #read in config and set class variable
        config = json.load(config_file_pointer)
        config_file_pointer.close() 
        try:
            assert float(config["db_sample"]) not in [0.0,1.0]
            assert float(config["pred_sample"]) not in [0.0,1.0]
            assert float(config["var_sample"]) not in [0.0,1.0]
        except AssertionError as error:
            print ("Bad configuration file, exiting program ...")
            print ({error},"occured\n")
            exit()
        Prover.config = config

    @staticmethod
    def check_consistent(clause_predicates,atom_lists,exists=False):
        """
        ensures that a variable is bound to only a single atom in a model
        """
        n_predicates = len(clause_predicates)
        consistent_models = list()

        for atom_config in atom_lists:
            inconsistent = False
            clause_arguments = {}

            for i in range(n_predicates):
                clause_predicate_args = clause_predicates[i].strip().split('(')[-1][:-1].split(',')
                for clause_predicate_arg in clause_predicate_args:
                    clause_arguments[clause_predicate_arg] = set()

            for i in range(n_predicates):
                clause_predicate_args = clause_predicates[i].strip().split('(')[-1][:-1].split(',')
                relevant_atom_args = atom_config[i].strip().split('(')[-1][:-1].split(',')
                n_args = len(clause_predicate_args)
                for j in range(n_args):
                    clause_arguments[clause_predicate_args[j]].add(relevant_atom_args[j])

            for clause_argument in clause_arguments:
                if len(clause_arguments[clause_argument]) > 1:
                    inconsistent = True
                for clause_argument2 in clause_arguments:
                    if not clause_argument2 == clause_argument and [element for element in clause_arguments[clause_argument2] if element in clause_arguments[clause_argument]]:
                        inconsistent = True

            if not inconsistent:
                consistent_models.append((atom_config,clause_arguments))
                if exists:
                    return True

        if not consistent_models:
            return False
        
        return consistent_models

    @staticmethod
    def is_same_signature(predicate1, predicate2):
        """
        checks if two predicates have the same signature, i.e., name and arity
        this is checked to allow valid unification
        """
        predicate1_name = predicate1.split('(')[0].strip() #get names
        predicate2_name = predicate2.split('(')[0].strip()
        n_predicate1_arguments = len(predicate1.strip().split('(')[-1][:-1].split(',')) #get arities
        n_predicate2_arguments = len(predicate2.strip().split('(')[-1][:-1].split(','))

        #if name and arities match return True, else return False
        if predicate1_name == predicate2_name and n_predicate1_arguments == n_predicate2_arguments:
            return True
        return False
    
    @staticmethod
    def ground_clause(clause,database,exists=False):
        """
        returns models for provable clause (returns True if exists = True),
        else if not provable, returns False
        """
        clause = clause.replace(' ','') #remove redundant spacing
        groundings = {'head': {}, 'body': {}} #initialize placeholder for ground model(s) atoms
        head_predicates = clause.split(':-')[0].split(';') #get head predicates
        body_predicates = clause.split(':-')[-1].split(';') #get body predicates

        #initialize place holders for head and body predicate groundings
        for head_predicate in head_predicates:
            groundings['head'][head_predicate] = []

        for body_predicate in body_predicates:
            groundings['body'][body_predicate] = []

        #get groundings of head predicates from the database 
        for head_predicate in head_predicates:
            for predicate in database:
                if Prover.is_same_signature(head_predicate,predicate):
                    groundings['head'][head_predicate] += [predicate]

        #get groundings of body predicates from the database
        for body_predicate in body_predicates:
            for predicate in database:
                if Prover.is_same_signature(body_predicate,predicate):
                    groundings['body'][body_predicate] += [predicate]

        #initialize placeholder for ground atoms
        atom_lists = list()

        #append ground head atoms to atom list
        for item in groundings['head']:
            atom_lists += [groundings['head'][item]]

        #append ground tail atoms to atom list
        for item in groundings['body']:
            atom_lists += [groundings['body'][item]]

        #combine head and body atom lists and return consistent models
        clause_predicates = head_predicates + body_predicates
        atom_lists = list(product(*atom_lists))
        consistent_models = Prover.check_consistent(clause_predicates,atom_lists,exists)
        return (clause,consistent_models)

    @staticmethod
    def prove_clause(clause,database):
        """
        attempts to ground the clause using atoms in the database, 
        if grounding returns a model, clause is proved,
        else clause is not proved, returns False
        """
        if not clause or not database: #no clause or database provided
            return False
        models = Prover.ground_clause(clause,database)
        if models[1]:
            return models[1]
        else:
            return models[1] #False case
        
    @staticmethod
    def get_predicate_signatures(db):
        """
        returns name and arities of predicates in the database (denoted as db)
        """
        predicate_names = [db_item.split('(')[0].strip() for db_item in db]
        predicate_set = set(predicate_names)
        predicate_signatures = {}
        for predicate in predicate_set:
            predicate_example = db[predicate_names.index(predicate)]
            n_predicate_args = len(predicate_example.strip().split('(')[-1][:-1].split(','))
            predicate_signatures[predicate] = n_predicate_args

        return predicate_signatures
        
    @staticmethod
    def seggregate(divisions,li):
        """
        seggregate list into buckets corresponding to predicate arities
        """
        divided_list = []
        for item in divisions:
            divided_list.append(li[:item]); li = li[item:]

        return divided_list

    @staticmethod
    def var_minimize(li):
        """
        remove redundant variable usage
        """
        c = 0
        min_li = [(0,0)]
        for item in li:
            if item not in [y[0] for y in min_li]:
                min_li.append((item,'V'+str(c))); c+= 1
            elif item in [y[0] for y in min_li]:
                var = [y[1] for y in min_li if item == y[0]][0]
                min_li.append((item,var))
        return [item[1] for item in min_li[1:]]
        
    @staticmethod
    def get_candidate_clauses(target, predicate_signatures):
        """
        returns candidate clauses based on target and database predicate signatures
        """
        target_name = target.split('/')[0] #get target name
        n_target_arguments = int(target.split('/')[1]) #get target arity
        predicate_name_combinations = [] #initialize place holder for clause body
        for i in range(Prover.config['max_clause_length']): #construct predicate variable combinations constrained to clause length
            predicate_name_combinations += list(product(list(predicate_signatures.keys()),repeat=i+1))
        candidate_clauses = set() #place holder for candidate clauses

        #sub sample for efficiency
        sub_sampled_predicate_name_combinations = sample(predicate_name_combinations, int(Prover.config["pred_sample"]*len(predicate_name_combinations)))
        if sub_sampled_predicate_name_combinations:
            predicate_name_combinations = sub_sampled_predicate_name_combinations

        #for each predicate variable combination, construct a clause
        print ("Generating candidate clauses ... \n")
        for combination in tqdm(predicate_name_combinations):
            
            #get number of variables and arities for variabalization
            n_args = [predicate_signatures[predicate] for predicate in combination]
            n_vars = n_target_arguments + sum(n_args)
            vars = ['V'+str(i) for i in range(n_vars)]

            #reduce redundant combinations
            var_combinations = [list_item for list_item in list(product(vars, repeat=n_vars))]
            sub_sampled_var_combinations = sample(var_combinations, int(Prover.config["var_sample"]*len(var_combinations))) #sub sample for efficiency
            if sub_sampled_var_combinations:
                var_combinations = sub_sampled_var_combinations
            var_combinations = [Prover.var_minimize(var_combination) for var_combination in tqdm(var_combinations)]

            #construct candidateclauses
            for var_combination in var_combinations:
                args = [str(item).replace('\'','').replace('[','(').replace(']',')') for item in Prover.seggregate([n_target_arguments]+n_args,var_combination)]
                candidate_clause_predicates = [''.join(list(item)) for item in list(zip([target_name]+list(combination),args))]
                candidate_clause = candidate_clause_predicates[0]+' :- '+'; '.join(candidate_clause_predicates[1:])
                arg_vars = sum([item[1:-1].split(',') for item in args],[])
                if arg_vars[0] in arg_vars[1:]: #inductive bias, can remove if not wanted
                    candidate_clauses.add(candidate_clause)

        #return constructed clauses
        return candidate_clauses
        
    @staticmethod
    def generate_candidate_clauses(target,database):
        """
        generates candidate clauses for proving target given,
        target signature, and facts in the database
        """
        db = database #shorthand

        #remove target predicates from db
        db_without_target = [db_item for db_item in db if not db_item.split('(')[0] == target.split('/')[0]]
        predicate_signatures = Prover.get_predicate_signatures(db_without_target) #get predicate signatures

        #get candidate clauses based on predicate signatures and target, and return
        candidate_clauses = Prover.get_candidate_clauses(target, predicate_signatures)
        return candidate_clauses
    
class Gradient_Approximation(object):
    """
    a decision stump for now, 
    can be any function approximator
    """
    def __init__(self,
                 learner,
                 knowledge_clauses = None,
                 default = None):
        """
        constructor that sets up placeholders for clause, and value; and initializes default value
        """ 
        self.clause, self.value = None, 0.0
        self.default = default #default value
        self.gradients = {} #initialize gradients
        self.knowledge_clauses = knowledge_clauses #add knowledge clauses

        for target in learner.gt_targets:
            self.gradients[target] = None

    def set_clause_and_value(self,
                             clause = None,
                             value = None):
        """
        sets clause and value for evaluation
        """
        self.clause = clause #set clause
        self.value = value #set clause value

    def evaluate(self,
                 target,
                 db):
        """
        returns a value based on match with clause or failure to match
        """

        models = Prover.prove_clause(self.clause,db)
        knowledge_models = [Prover.prove_clause(knowledge_clause,db) for knowledge_clause in self.knowledge_clauses]
        knowledge_models = [models for models in knowledge_models if models]
        knowledge_value = 0.0
        if knowledge_models:
            knowledge_value = target[0] in [item[0][0] for item in sum(knowledge_models,[])]
            if knowledge_value:
                knowledge_value = 1.0
            else:
                knowledge_value = -1.0
        if not models:
            return self.default + knowledge_value
        model_targets = [item[0][0] for item in models]
        return [self.value + knowledge_value if target in model_targets else self.default for target in [target]][0]

    def __repr__(self):
        """
        returns string representation of class object
        """  
        return str(self.__dict__)
    
class KiRL(object):
    """"
    implements various knowledge infused reinforcement learning methods
    """

    def __init__(self, 
                 ideal = True,
                 data_object = None):
        """
        constructor
        """
        self.config = None #setup trainer using trainer_config.json
        f = open('trainer_config.json'); self.config = json.load(f); f.close()
        if ideal:
            self.candidate_clauses = data_object.get_ideal_candidate_clauses() #set candidate clauses
            self.knowledge_clauses = data_object.get_knowledge_clauses() #set knowledge clause
        self.db = data_object.get_db() #set database of facts, positive and negative examples
        self.action_set = data_object.get_action_set() #get set of possible actions

        self.target = data_object.get_target() #set target(s)
        self.gt_targets = {} #set ground truth values for targets
        for item in data_object.positive_targets:
            self.gt_targets[item] = 1.0
        for item in data_object.negative_targets:
            self.gt_targets[item] = 0.0

        def sigmoid(x): #define activation function for squashing regr values
            """
            sigmoid function
            """
            return exp(x)/float(1+exp(x))
        
        self.activation = sigmoid
        #initialize model as a set of functional gradient approximations
        self.model = [Gradient_Approximation(self,
                                             knowledge_clauses = self.knowledge_clauses,
                                             default=self.config["default"]) for _ in range(self.config["n_steps"]+1)]

    def get_var_and_mean(self,gradients,clause):
        """
        scores a given clause based on weighted sample variance
        """
        #setup data structures
        models = Prover.ground_clause(clause,self.db) #get clause models
        if not models[1]: #if no models return gradient = sigmoid(0)-1, and variance = max from config file
            return self.config["max_var"], -0.5
        model_targets = [item[0][0] for item in models[1]] #else obtain target gradients
        parent_list = list(gradients.keys())
        target_values = [gradients[target] for target in parent_list if target in model_targets]

        #score clause
        n_parent_list = len(parent_list) #score clause based on sample variance
        if not target_values or len(target_values) == 1:
            return float(n_parent_list), -0.5
        n_target_values = len(target_values)
        mean_value = sum(target_values)/float(n_target_values)
        var = sum([(value - mean_value)**2 for value in target_values])/float(n_target_values-1) #sample variance
        return var, mean_value #return clause mean, and calculated sample variance
        
    def imitation_learn(self):
        """
        trains a classifier that uses a sum of approximate functional graidents
        """
        targets = list(self.gt_targets.keys()) #set ground truth targets
        gradient_step_approximations = self.model #shorthand for learned model
        print ("%% training model with "+str(self.config["n_steps"])+" gradient steps ... ")
        for step in tqdm(range(1,self.config["n_steps"]+1)): #learn each gradient approximation step
            target_values, gradients = {}, {}
            for target in targets: #(1) for each target compute value as sum of previous gradient steps
                target_values[target] = sum([approx.evaluate(target,self.db) for approx in gradient_step_approximations[:step]])
                gradients[target] = self.gt_targets[target] - self.activation(target_values[target])
            gradient_step_approximations[step].gradients = gradients #set current gradient as target - sum in (1)
            #obtain best clause based on lowest sample variance across gradient values
            clauses_stats = [self.get_var_and_mean(gradients,clause) for clause in self.candidate_clauses]
            min_index = [clauses_stat[0] for clauses_stat in clauses_stats].index(min([clauses_stat[0] for clauses_stat in clauses_stats]))
            best_clause, best_clause_value = self.candidate_clauses[min_index], clauses_stats[min_index][-1]
            #set best clause in current gradient approximation
            gradient_step_approximations[step].set_clause_and_value(best_clause,best_clause_value)

    def predict(self,context):
        """
        predicts a next action based on learned model, and given set of facts
        time = the current time step
        context = partial or full observation/state history until the current time step
        """
        time = max([int(item.split(',')[-1][:-1][1:]) for item in context])+1
        action_set = self.action_set #shorthand for action set

        if context is None: #if no context provided, every action is equally good
            return choice(list(action_set.keys())).split(')')[0]+',t'+str(time)+')' #return randomly chosen action
        
        for action in action_set: 

            #format timed action and timed target
            timed_action = action.split(')')[0]+',t'+str(time)+')'
            timed_target = self.target.split('/')[0]+'('+str(action_set[action])[1:-1].replace('\'','')+',t'+str(time)+')'
            db = context+[timed_action, timed_target]
            db += ['next(t'+str(time-1)+',t'+str(time)+')']
            #for t in range(time):
            #    next_step_predicate = 'next(t'+str(t)+',t'+str(time)+')'
            #    db += [next_step_predicate]
            value = sum([approx.evaluate([timed_target],db) for approx in self.model])
            return value          

    def __repr__(self):
        """
        returns stirng representation of objects of this class
        """
        return str(self.__dict__)
