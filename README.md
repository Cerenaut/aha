# memory
Selective memory system based on convolutional, capsular/modular hierarchical k-sparse autoencoders


## Testing
Unit tests can be executed in multiple ways:

1. Execute an individual test: `python -m components.autoencoder_component_test`
2. Execute all tests in a specific directory: `python -m unittest discover -s ./components -p '*_test.py'`
3. Execute all the tests in the project: `python -m unittest discover -p '*_test.py'`