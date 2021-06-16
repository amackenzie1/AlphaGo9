Here we have 9x9 Go using the AlphaZero Monte Carlo tree search and self-play algorithm. Amazingly enough, it works; it beat a 9-kyu bot on OGS (playing black with komi of 5.5). 

It still makes (occasionally rather obvious) mistakes when it's losing, and has some issues with long-term life-and-death planning, but otherwise plays pretty decently. 

Game generation ran on a cluster of 84 CPUs at McGill; training happened in parallel on a single GPU. The number of bugs that this algorithm can tolerate and still "somewhat work" is absolutely unbelievable; I caught two or three mistakes each time I moved from Connect4 to 5x5 Go, from 5x5 Go to 7x7 Go, and from 7x7 Go to 9x9 Go. 

The requirements here are tensorflow, scikit-learn, and matplotlib; to play against the network, just run python play.py. 
