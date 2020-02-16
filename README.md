Hanabi Framework (HF) is a convenience package to run a game of hanabi with RL agents.


The main class is `HanabiGameManager`. It consists of one environment and a list of agents and makes sure that the game proceeds as intended and agents make their moves and get all necessary information. It aims at providing a deeper integration between agents and the env, thus achieving better performance. However, it also means that the manager losses in generality, makes certain assumptions about agents, which those have to compily with. These assumptions are formalized in the form of a `HanabiAgent` class.
