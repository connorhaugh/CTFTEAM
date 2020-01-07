# CTFTEAM

![Image of Yaktocat](/pacman_cropped.png)

My Group's submission to Amherst's AI capture the flag tournament in the fall of 2019. The challenge was to make two agents which would compete in a Capture the flag tourament, where the flags were pellets, and agents were ghosts (concerned with eating pacmen) in thier own half, and pacmen trying to eat pellets and bring them back to thier half. The Team, contained in myTeam.py contians two agents:
# OnePac


OnePac is the "attacking agent," using a combination of q-learned feature weights and particle filtering to estimate the position of potential enemies. OnePac is the subject I spent the most time working on.
# TwoPac
TwoPac perfers to play defense, although whe the scores are desperate, he sometimes will be forced to go steal some pellets. It uses a little adversarial search to nab nere-do-well pacmen who try to steal his pellets. It also has a few hard-coded "set-plays" to deal with situations which seem logical, but may help in the long run.
