# Knowledge-infused Reinforcement Learning Development Repository

This repository contains implementations for knowledge-infused reinforcement learning methods

## Library Installations
```
pip install -r requirements.txt
```

## Execution Commands
```
python main.py
```

## Example Execution
```
from Classes import Synth_Data, Unit_Tests, Prover, KiRL

def run():
    """
    main method
    """
    Prover.set_config() #set prover configuration from config.json file, e.g., max clause length
    synthetic_data_obj = Synth_Data(dataset="stock_trading") #create synthetic dataset

    #vanilla imitation learning (training)
    learner = KiRL(data_object=synthetic_data_obj)
    learner.imitation_learn()
    print ("%% The learned model and gradient approximations are ... ")
    print (learner.model)
    print ('='*40)

    #inference using the learned model
    test_context, time =synthetic_data_obj.get_test_facts()
    predicted_value = learner.predict(test_context)
    print (predicted_value)

if __name__ == '__main__':
    run()
```
