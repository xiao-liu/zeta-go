# Overview
ZetaGo is a python implementation of Deepmind's AlphaGo Zero.
I choose the greek letter "zeta" as the name because of its similarity to the
word "zero".
ZetaGo is almost an exact reproduction of AlphaGo Zero, with some minor
differences (see Section "Differences from AlphaGo Zero" for a complete list).
Through runnable, the nature of this project is more demonstrative than
practical.
A rough estimation by myself shows that it would take over 30,000 years to
finish the training on my own desktop
(CPU: Intel Core i5-6600K / GPU: NVidia GTX 1080 / RAM: DDR4 16GB).

Given the infeasibility of training a standard AlphaGo Zero model in my
computer, I turn to solving a smaller problem.
In particular, I attempt to train a good model for 9x9 Go board.

I have taken efforts to make ZetaGo as self-contained as possible.
It has minimal external dependencies: `pytorch` for machine learning framework,
`numpy` for matrix manipulation, `pygame` for GUI and `glog` for logging.

# Choice of Rules of Go
As is well known, there exist several variants for the rules of Go.
The differences mainly come from the following five aspects:

* Scoring: ZetaGo adopts area scoring.
Territory scoring introduces way too many unnecessary concepts and definitions
which make it rather complicated and less intuitive.
Compared to territory scoring, area scoring is much more concise and beautiful.
AlphaGo Zero adopts area scoring as well.

* Ko: ZetaGo only considers simple ko.
In contrast, AlphaGo Zero considers super ko.
Unlike my strong preference to area scoring than territory scoring, I basically
have a neutral attitude to simple ko and super ko.
I make this decision for several reasons:

  * I slightly feel that super ko might be an overly strong restriction, and
  from the point of mathematical elegance, I prefer to introduce as few
  unnatural restrictions as possible.
  Super ko forbids a position as long as it appears before, without considering
  the path (i.e., history of playing) that leads to it.
  Of course, I also admit that simple ko might be too weak.
  It only prevents immediate repeating play, but does not exclude the
  possibility of repetitions with longer cycles.

  * There are situational super ko, positional super ko and natural situational
  super ko.
  I do not know which one to choose...

  * (Perhaps the most important reason) Simple ko is very cheap to implement as
  we only need to remember the playing of the last step.
  To implement super ko, the entire playing history must be stored.

* Suicide: ZetaGo forbids suicide.
Again I have a neutral attitude toward this.
In fact, based on my philosophy of "fewer restrictions", I should even prefer
allowing suicide.
However I still choose to forbid it because in practice suicidal moves are
mostly stupid and allowing suicide will make your algorithm spend time on
exploring these useless moves.
If you really want to allow suicidal moves, you can enable it by just editing a
few lines of code.
AlphaGo Zero does not explicitly specify their choice.

* Handicap: We always assume no handicap stones when training the model,
therefore the rule of placing handicap stones does not matter.
The underlying `Go` class allows placing handicap stones anywhere, instead of
just on a set of predefined locations.

* Komi: We choose komi=7.5.

# Differences from AlphaGo Zero
* Since ZetaGo only considers simple ko, a history of length 3 is sufficient to
fully determine the state and hence the input of ZetaGo only has 7 (=2\*3+1)
feature planes.
As a comparison, AlphaGo Zero considers a history of length 8 and it has 17
(=2\*8+1) feature planes.

* The optimization of ZetaGo's neural network is single-threaded.
As a result, virtual loss trick is no longer needed and I do not implement it.
In AlphaGo Zero, the three components (the optimization of the neural network,
the comparison of the neural networks, and the generation of self-play data) are
asynchronously executed in parallel.
In ZetaGo, because it is single-threaded, the three components run sequentially.

  The generation of self-play data and the comparison of the neural networks
themselves support multi-processing.
These two components are relatively easy to parallelize because different runs
of self-play / network comparison are independent.
In addition, paralleling the runs also allows us to submit a batch of neural
network evaluation requests to GPU at once, which improves the I/O throughput.

* The description of AlphaGo Zero's resignation system lacks details.
In the paper, the authors wrote:
"AlphaGo Zero resigns if its root value and best child value are lower than a
threshold value v_resign."
However they never defined what "root value" and "child value" are.
The description of resignation in AlphaGo is much clearer, in which they
explicitly used quantity max_a{Q(s, a)} (the best action value of the root s) to
determine resignation.
I believe this is also the definition of root value for AlphaGo Zero, and the
child value is just the best action value of the child node in question.
(But remember to negate the child value before you use it, as it is calculated
from the opponent's perspective. Actually you need to repeatedly negate the
value at the backup stage of MCTS as well, and the paper does not emphasize
this. See the code at the end of `mcts.py`.)

# Usage
ZetaGo provides three basic functions:
* Train a new model;
* Resume training from a previous checkpoint;
* Play Go against computer with a specified model.

The entry point is `main.py`.
You can type ```python main.py --help``` to show the help message, and type
```python main.py <command> [args]...``` to execute a command.
Currently three commands are supported: `train`, `resume` and `play`, which is
used to train a new model, to resume training from a previous checkpoint, and to
play Go against computer with a specified model, respectively.
You can also type ```python main.py <command> --help``` to show the help message
for each command.

**Train a new model**

Run ```python main.py train``` to start training a new model.
By default, ZetaGo will train a standard AlphaGo Zero model.
You can train a different model with different parameters by specifying a
different configuration using the `--config` flag.
All the configurations are defined in `config.py`.
I have provided two configurations: `19x19`, which is the standard setup of
AlphaGo Zero, and `9x9`, which corresponds to a smaller setup of 9x9 Go board.

You may also specify a name for your model using the `--model_name` flag.
If not specified, ZetaGo will use timestamp as name.
All the checkpoints and the final model will be saved to the directory
`models/<model_name>`.

You can use the `--num_workers` flag to specify the number of processes during
the generation of self-play data and the comparison of the networks.
By default it is 4 times the number of CPUs in your computer.

**Resume training from a previous checkpoint**

Run ```python main.py resume <model_name>``` to resume the training.
By default ZetaGo will resume from the latest checkpoint.
You can specify a different checkpoint to resume from using the `--checkpoint`
flag.
Use the `--config` flag to specify a new configuration and override the one
saved in the checkpoint (make sure that they are compatible).
The `--num_workers` flag is also supported here.

**Play Go against computer with a specified model**

Run ```python main.py play <model_name>``` to play Go against computer with the
specified model.
By default human is the black player, and this can be changed using the
`--black_player` flag.
You can run the following python script in the `src` directory to create a
random neural network:
```python
import torch
from config import get_conf
from network import ZetaGoNetwork

conf = get_conf('19x19')
network = ZetaGoNetwork(conf)

torch.save(
    {
        'conf': conf,
        'best_network': network.state_dict(),
    },
    '../models/19x19_random/model.pt',
)
```
Then you can try the GUI by typing `python main.py play 19x19_random`.
Do not laugh if it makes silly moves!


# Structure of Source Code Files
The implementation of ZetaGo is compact.
It only contains a dozen of files and most of them have only 100~200 lines.
Therefore I choose a flat file structure and put all the files directly under
the `src` directory.
Following is an introduction to each file:

* `compare.py`

  The code that compares the performance of two neural networks.

* `config.py`

  The collection of configurations.
  A configuration contains all the information that defines the problem and all
  the parameters that are needed to train the model.

* `data_structure.py`

  Implementations of some basic data structures like set and queue.
  I re-implement them for better performance.
  You can also use their build-in counterparts if you like.

* `evaluate.py`

  Implementations of two evaluators which calculate the outputs of neural
  networks for given inputs.
  The `DefaultEvaluator` is just a wrapper of the underlying neural network.
  The `BulkEvaluator` supports multiple processes to submit evaluation requests
  to multiple neural networks.
  Inside `BulkEvaluator`, it will pack e requests from different processes
  together and submit them to the corresponding neural network at once.
  This greatly improves the I/O throughput.

* `example.py`

  The class that generates and manages self-play examples.

* `go.py`

  The implementation of Go rules.
  Class `Board` manages the static information about the board such as stone
  chains and liberties.
  Class `Go` manages the dynamic information about a game, such as players' turn
  and ko.

* `gui.py`

  A simple graphic user interface.

* `main.py`

  Entry point of ZetaGo.

* `mcts.py`

  An implementation of Monte Carlo tree search.

* `network.py`

  The definition of the neural network.

* `play.py`

  All the code related to playing Go games, including self-play, computer v.s.
  computer and human v.s. computer.

* `predict.py`

  The code that calculates the prediction of a neural network for a given input,
  applying a random Dihedral transformation if necessary.

* `resign.py`

  The code that calculates the threshold for resignation based on recent
  self-play history.

* `train.py`

  The code to train a new model/resume training from a previous checkpoint.
