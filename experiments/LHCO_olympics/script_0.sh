#!/bin/bash
python lhco_flowmatching.py 512 0.001 512 1 0 OptimalTransportFlowMatching
python lhco_flowmatching.py 512 0.001 512 1 1 OptimalTransportFlowMatching
python lhco_flowmatching.py 1024 0.0001 512 1 0 OptimalTransportFlowMatching
python lhco_flowmatching.py 1024 0.0001 512 1 1 OptimalTransportFlowMatching