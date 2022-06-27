# Argument quality scoring

First, load the data from [[1]](#Gretz.2020) as described in the paper and split them into training
and test set accordingly. 

`main.py` resembles the proposed training scenario by [[1]](#Gretz.2020). Run the script to obtain the topic 
independent model. 
To evaluate the model on the test set, execute `inference.py` script.

---
<span id="Gretz.2020">[1] Shai Gretz, Roni Friedman, Edo Cohen-Karlik, Assaf Toledo, Dan Lahav, Ranit
Aharonov, and Noam Slonim. 2020. A large-scale dataset for argument quality
ranking: Construction and analysis. _In Proceedings of the AAAI Conference on
Artificial Intelligence_, Vol. 34. 7805–7813.</span>