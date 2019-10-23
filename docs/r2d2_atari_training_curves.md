# Training curves of SEED's R2D2 on ATARI games.

Below are training curves for SEED's R2D2 on Atari-57. We trained on 4 seeds up
to 40e9 frames. The curves represent the mean undiscounted returns over the 4
seeds. The shaded area around the curves represent a 95% confidence interval for
the mean computed using bootstrapping.

The hyperparameters and evaluation procedure are the same as in section A.3.1 in
the [paper](https://arxiv.org/pdf/1910.06591.pdf). We highlight that we use up
to 30 random no-ops to start the episodes, no sticky actions and that we stop
episodes after 30 minute (108K frames).

The slight differences compared to the training curves in the paper are due to a
few differences between the runs of the paper and the ones presented here:

-  **Emulator.** For the paper we used the
   [xitari](https://github.com/deepmind/xitari) ATARI 2600 emulator, whereas
   this run uses the emulator included in
   [atari_py](https://github.com/openai/atari-py/tree/master/atari_py).

- **Resizing.** For the paper, we used bilinear resizing provided in
  [PIL](http://www.pythonware.com/products/pil) which is not compatible
  with python3. For this run, we switched to OpenCV's bilinear resizing.

- **Ordering of pre-processing operations.** For the paper, we max-pool consecutive
  observations before grayscaling as in the [Human-level control through deep
  reinforcement learning](https://www.nature.com/articles/nature14236) paper.
  For this run, we switched to the environment preprocessing performed by
  [Dopamine](https://github.com/google/dopamine) which grayscales first and
  max-pools consecutive observations second.

Source data for all curves below can be downloaded in CSV format [here](seed_r2d2_atari_graphs.csv).

## Training curves

| ![Training curve of SEED's R2D2 On Alien](r2d2_atari_training_curves/Alien.png) | ![Training curve of SEED's R2D2 On Amidar](r2d2_atari_training_curves/Amidar.png) |
| ----- | -----|
| ![Training curve of SEED's R2D2 On Assault](r2d2_atari_training_curves/Assault.png) | ![Training curve of SEED's R2D2 On Asterix](r2d2_atari_training_curves/Asterix.png) |
| ![Training curve of SEED's R2D2 On Asteroids](r2d2_atari_training_curves/Asteroids.png) | ![Training curve of SEED's R2D2 On Atlantis](r2d2_atari_training_curves/Atlantis.png) |
| ![Training curve of SEED's R2D2 On BankHeist](r2d2_atari_training_curves/BankHeist.png) | ![Training curve of SEED's R2D2 On BattleZone](r2d2_atari_training_curves/BattleZone.png) |
| ![Training curve of SEED's R2D2 On BeamRider](r2d2_atari_training_curves/BeamRider.png) | ![Training curve of SEED's R2D2 On Berzerk](r2d2_atari_training_curves/Berzerk.png) |
| ![Training curve of SEED's R2D2 On Bowling](r2d2_atari_training_curves/Bowling.png) | ![Training curve of SEED's R2D2 On Boxing](r2d2_atari_training_curves/Boxing.png) |
| ![Training curve of SEED's R2D2 On Breakout](r2d2_atari_training_curves/Breakout.png) | ![Training curve of SEED's R2D2 On Centipede](r2d2_atari_training_curves/Centipede.png) |
| ![Training curve of SEED's R2D2 On ChopperCommand](r2d2_atari_training_curves/ChopperCommand.png) | ![Training curve of SEED's R2D2 On CrazyClimber](r2d2_atari_training_curves/CrazyClimber.png) |
| ![Training curve of SEED's R2D2 On DemonAttack](r2d2_atari_training_curves/DemonAttack.png) | ![Training curve of SEED's R2D2 On DoubleDunk](r2d2_atari_training_curves/DoubleDunk.png) |
| ![Training curve of SEED's R2D2 On Enduro](r2d2_atari_training_curves/Enduro.png) | ![Training curve of SEED's R2D2 On FishingDerby](r2d2_atari_training_curves/FishingDerby.png) |
| ![Training curve of SEED's R2D2 On Freeway](r2d2_atari_training_curves/Freeway.png) | ![Training curve of SEED's R2D2 On Frostbite](r2d2_atari_training_curves/Frostbite.png) |
| ![Training curve of SEED's R2D2 On Gopher](r2d2_atari_training_curves/Gopher.png) | ![Training curve of SEED's R2D2 On Gravitar](r2d2_atari_training_curves/Gravitar.png) |
| ![Training curve of SEED's R2D2 On Hero](r2d2_atari_training_curves/Hero.png) | ![Training curve of SEED's R2D2 On IceHockey](r2d2_atari_training_curves/IceHockey.png) |
| ![Training curve of SEED's R2D2 On Jamesbond](r2d2_atari_training_curves/Jamesbond.png) | ![Training curve of SEED's R2D2 On Kangaroo](r2d2_atari_training_curves/Kangaroo.png) |
| ![Training curve of SEED's R2D2 On Krull](r2d2_atari_training_curves/Krull.png) | ![Training curve of SEED's R2D2 On KungFuMaster](r2d2_atari_training_curves/KungFuMaster.png) |
| ![Training curve of SEED's R2D2 On MontezumaRevenge](r2d2_atari_training_curves/MontezumaRevenge.png) | ![Training curve of SEED's R2D2 On MsPacman](r2d2_atari_training_curves/MsPacman.png) |
| ![Training curve of SEED's R2D2 On NameThisGame](r2d2_atari_training_curves/NameThisGame.png) | ![Training curve of SEED's R2D2 On Phoenix](r2d2_atari_training_curves/Phoenix.png) |
| ![Training curve of SEED's R2D2 On Pitfall](r2d2_atari_training_curves/Pitfall.png) | ![Training curve of SEED's R2D2 On Pong](r2d2_atari_training_curves/Pong.png) |
| ![Training curve of SEED's R2D2 On PrivateEye](r2d2_atari_training_curves/PrivateEye.png) | ![Training curve of SEED's R2D2 On Qbert](r2d2_atari_training_curves/Qbert.png) |
| ![Training curve of SEED's R2D2 On Riverraid](r2d2_atari_training_curves/Riverraid.png) | ![Training curve of SEED's R2D2 On RoadRunner](r2d2_atari_training_curves/RoadRunner.png) |
| ![Training curve of SEED's R2D2 On Robotank](r2d2_atari_training_curves/Robotank.png) | ![Training curve of SEED's R2D2 On Seaquest](r2d2_atari_training_curves/Seaquest.png) |
| ![Training curve of SEED's R2D2 On Skiing](r2d2_atari_training_curves/Skiing.png) | ![Training curve of SEED's R2D2 On Solaris](r2d2_atari_training_curves/Solaris.png) |
| ![Training curve of SEED's R2D2 On SpaceInvaders](r2d2_atari_training_curves/SpaceInvaders.png) | ![Training curve of SEED's R2D2 On StarGunner](r2d2_atari_training_curves/StarGunner.png) |
| ![Training curve of SEED's R2D2 On Tennis](r2d2_atari_training_curves/Tennis.png) | ![Training curve of SEED's R2D2 On TimePilot](r2d2_atari_training_curves/TimePilot.png) |
| ![Training curve of SEED's R2D2 On Tutankham](r2d2_atari_training_curves/Tutankham.png) | ![Training curve of SEED's R2D2 On UpNDown](r2d2_atari_training_curves/UpNDown.png) |
| ![Training curve of SEED's R2D2 On Venture](r2d2_atari_training_curves/Venture.png) | ![Training curve of SEED's R2D2 On VideoPinball](r2d2_atari_training_curves/VideoPinball.png) |
| ![Training curve of SEED's R2D2 On WizardOfWor](r2d2_atari_training_curves/WizardOfWor.png) | ![Training curve of SEED's R2D2 On YarsRevenge](r2d2_atari_training_curves/YarsRevenge.png) |
| ![Training curve of SEED's R2D2 On Zaxxon](r2d2_atari_training_curves/Zaxxon.png) | |

