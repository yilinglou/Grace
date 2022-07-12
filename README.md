
## Grace: Graph-based representation learning for fault localization

Grace is a coverage-based fault localization approach with graph-based representation. This repository includes data and code about the technique.


###  Organization

    .
	|-- Code 
		|-- Default 
		|-- Variants
			|-- Coarse_code 
			|-- Coarse_test
			|-- Loss_pair
			|-- Loss_point



This directory presents main implementation of Grace, including the default version, and four variant versions.

* *Default*  presents the implementation of default Grace, including the code on model and code on input graph construction.

* *Variants*  presents four variants in terms of graph representations and ranking loss.
    * *Loss.* Replace list loss with pair and point loss.
    * *Representation.* Replace fine-grained test and code representations with coarse-grained ones.


### Environment
```
PyTorch: V1.7.1
OS: Ubuntu 16.04.6 LTS
```

### Execution
runtotal.py is  main entry file. 


