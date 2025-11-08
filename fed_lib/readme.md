# fed_lib — Design Rationale and Developer Notes

This README focuses on *why* the modules and functions in `fed_lib` are designed the way they are. It assumes you can read each file for the exact implementation; here we explain the design trade-offs, engineering intent, and the assumptions that drove the code layout.

## High-level goals

- Keep the federation simulation simple, reproducible, and easy to extend.
- Favor clear, minimal building blocks (small model, single-epoch client training, explicit gradient aggregation) so different federated algorithms can be experimented with without complex, hidden machinery.
- Provide sensible defaults that make experiments stable and debuggable (SGD, CrossEntropy, stratified splits, reproducible shuffling).

The library intentionally emphasizes clarity over maximum performance or absolute realism (e.g., no differential privacy, no secure aggregation). Those features are natural future additions.

## `utils.py`

Overview: `utils.py` contains compact model definitions, training/evaluation helpers, and dataset partitioning utilities. The focus in design decisions was: (1) make the model lightweight for fast FL experiments, (2) make gradient bookkeeping explicit and correct when aggregating, and (3) provide reproducible, stratified splits for the homogeneous-domain experiments.

- Small model (`SmallConvBlock`, `SmallCNN`)
	- Why small? Federated experiments often need many client trainings and rounds. A small network runs quickly on modest hardware and highlights algorithmic differences (FedSGD vs centralized) without long training times.
	- Why BatchNorm + Dropout + AdaptiveAvgPool? BatchNorm stabilizes training across small batches (useful when client loaders have variable sizes). Dropout adds lightweight regularization to reduce overfitting at the client level. AdaptiveAvgPool reduces positional sensitivity and ensures a small fixed-size feature vector for the linear head, which keeps the head tiny and fast.

- Training helpers (`train_model_one_epoch`, `model_one_epoch_losses`)
	- Why scale loss by batch size and divide gradients by total samples before optimizer.step? Many FL aggregation approaches assume gradients are averaged across all samples that clients processed. The implementation computes per-batch backward passes but accumulates a scaled loss (loss * batch_size) and finally divides parameter gradients by the total number of samples. This reproduces the exact gradient that would have been computed by summing per-example gradients and then averaging — which is important for correct weighting when clients process different batch sizes or different numbers of samples.
	- Why a single optimizer.step after a full pass (rather than stepping per batch)? For the centralized baseline we want the single-step equivalent of summing gradients across the epoch then applying the update once — matching the FedSGD aggregation semantics where clients compute epoch-aggregated gradients and the server applies the aggregated gradient once.
	- Why use `non_blocking=True`, `pin_memory`, `num_workers` propagation? These are standard performance options for moving CPU tensors to GPU efficiently. Forward/backward implementation remains device-agnostic; passing these flags keeps loader behavior consistent with how users commonly run PyTorch code.

- Evaluation helpers (`evaluate_model_on_test`)
	- Uses `torch.inference_mode()` for faster evaluation and to avoid building autograd graphs. Uses the same normalization pipeline as training to ensure valid metric comparisons.

- Data utilities (`get_cifar10`, `_make_stratified_subsets`, `get_homogenous_domains`)
	- Default transforms: a small augmentation (random crop, horizontal flip) for training but a deterministic normalization for test. This balances realistic training augmentation with a stable test set.
	- Stratified split rationale: To assess federated algorithms under homogeneous domains we want clients to have similar class distributions (rather than pathological non-IID splits). `_make_stratified_subsets` assigns indices per-class proportionally to each client so each client receives class-balanced subsets relative to its share of the dataset. This reduces confounding factors when comparing FedSGD to centralized training.
	- Why per-class allocation with fractional rounding? It preserves class balance per client. A naive uniform slicing can accidentally create class imbalances that dominate observed results.

## `fed_methods.py`

Overview: the `FedMethod` base class defines a concise interface: `exec_client_round`, `exec_server_round`, and evaluation hooks. `FedSGD` implements a straightforward, explicit gradient-aggregation federated algorithm.

- Minimal abstract interface (`FedMethod`)
	- Why an abstract class? This keeps the federation runner (`fed_model.Federation`) independent from the specific update rule. New algorithms (FedAvg, FedProx, etc.) can be added by implementing the same small interface.

- FedSGD implementation choices
	- Why aggregate gradients (not parameter deltas)? The code makes client models compute gradients with respect to a shared loss objective, then aggregates those gradients on the server and performs a single `optimizer.step()` on the server model. This matches the original FedSGD conceptual model where clients compute gradients and the server takes a step using an average gradient. Aggregating gradients directly is explicit and easy to reason about when comparing to classical centralized SGD.
	- Why `client.load_state_dict(server.state_dict())` at the start of client training? It synchronizes clients to the current server parameters. This matches synchronous FL rounds where clients start from the same model each round. It also ensures computed gradients are all in the same parameter reference frame so averaged gradients are meaningful.
	- Why client-level gradient normalization (dividing by total client samples)? Each client accumulates gradients across its local batches; dividing by the client's total samples produces the client's average gradient. Aggregating those averages with client weights produces the expected global average gradient.
	- Why support `client_weights` and weight normalization? Some federated schemes require weighting clients by dataset size or importance. The normalization helper ensures weights sum to 1, and falls back safely to equal weighting when nothing is specified. This explicitness makes experiments reproducible and avoids silent mis-weighting.
	- Why zero-grad with `set_to_none=True` on server optimizer? This is a modern PyTorch performance pattern that avoids extra memory writes and can be slightly faster. The code still works if the environment does not support that flag.
	- Why the debug gradient-norm prints? Gradient norms are a quick, low-cost diagnostic to confirm gradients were computed and aggregated correctly, which is vital when implementing and debugging custom aggregation.

## `fed_model.py`

Overview: `Federation` orchestrates clients, the central baseline model, and the chosen `FedMethod` algorithm. The design keeps the runner minimal and delegates algorithmic behavior to `FedMethod` implementations.

- Why keep both `server` and `central_model`? The server is the federated model updated via the federated method. `central_model` is a deep copy trained centrally as a baseline. Running both in parallel makes comparisons straightforward and deterministic within the same training loop and random seeds.

- Why `get_homogenous_domains` gives client dataloaders? This lets the `Federation` focus on orchestration instead of dataset partitioning logic; tests and experiments can swap different partitioning strategies (homogeneous / heterogeneous) without changing the runner.

- Why `train()` hands `kwargs` into the fed method? It provides flexibility: different methods need different runtime context (optimizer, test loader, device, verbosity). Passing them as kwargs avoids coupling the runner to specific method requirements.

## `compare_model_parameters` and `compute_model_difference` — why these diagnostics

- Why layer-wise L2 / L-inf / relative measures? Per-layer statistics help locate which parts of the model diverge most between federated and centralized training. Relative differences (L2 normalized by parameter norm) are useful when parameter magnitudes vary a lot across layers.
- Why present top-k layers? In many networks only a few layers change significantly; showing top-k focuses debugging efforts.

## Contracts (short)

- SmallCNN
	- Input: tensor (B, 3, 32, 32)
	- Output: logits (B, 10)
	- Error modes: mismatched input sizes will error at conv/linear layers.

- train_model_one_epoch
	- Inputs: model, train_loader, criterion, optimizer, device
	- Output: (avg_loss, accuracy)
	- Behavior: accumulates scaled losses, divides gradients by total samples, then performs a single optimizer.step()

- FedSGD.exec_client_round
	- Inputs: server, client list, client_dataloaders, device, verbose
	- Behavior: sync clients to server, compute per-client averaged gradients, expose `client_sizes` in kwargs for server aggregation

- FedSGD.exec_server_round
	- Inputs: clients, server, server_optimizer, optional client_sizes/weights
	- Behavior: weight-normalized gradient aggregation, applies server_optimizer.step()

## Edge cases and how they're handled

- Empty clients list: `exec_server_round` raises early so caller knows the setup is wrong.
- Clients with zero samples: `_train_client` returns zero samples and zero loss; the server aggregation logic tolerates missing gradients by skipping None grads.
- Different devices: the code ensures gradients are moved to the server parameter device before aggregation.

## Limitations and suggested next steps (future work)

- Heterogeneous (non-IID) domain generation: the placeholder `_make_heterogenous_subsets` exists; a Dirichlet-based sampler would be a typical addition.
- FedAvg vs FedSGD: FedAvg (averaging model parameters after local optimization) is a complementary method that tends to be more practical in standard FL research — add as a new `FedMethod` implementation.
- Optimizer choices and momentum: FedSGD uses plain SGD for clarity; experiments could add momentum, Adam-style server updates, or per-client optimizers.
- Privacy / Communication: add quantization, compression, secure aggregation, and differential privacy for more realistic FL setups.

## Quick usage notes (cursory)

- Create a federation with some clients, pick a method (e.g., `FedSGD()`), and call `train(rounds=...)` on the `Federation` object. The runner will print per-round diagnostics and compare the federated server with a centralized model trained in the same loop.