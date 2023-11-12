#!/bin/bash
python lhco_flowmatching.py 512 0.001 512 2 0 SchrodingerBridgeFlowMatching
python lhco_flowmatching.py 512 0.001 512 2 1 SchrodingerBridgeFlowMatching
python lhco_flowmatching.py 1024 0.0001 512 2 0 SchrodingerBridgeFlowMatching
python lhco_flowmatching.py 1024 0.0001 512 2 1 SchrodingerBridgeFlowMatching