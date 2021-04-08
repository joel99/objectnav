# Abstract
ObjectGoal Navigation (ObjectNav) is an embodied task wherein agents are to navigate to an object instance in an unseen environment. Prior works have shown that end-to-end ObjectNav agents that use vanilla visual and recurrent modules, e.g. a CNN+RNN, perform poorly due to overfitting and sample inefficiency. This has motivated current state-of-the-art methods to mix analytic and learned components and operate on explicit spatial maps of the environment. We instead re-enable a generic learned agent by adding auxiliary learning tasks and an exploration reward. Our agents achieve 24.5% success and 8.1% SPL, a 37% and 8% relative improvement over <a href="https://www.cs.cmu.edu/~dchaplot/projects/semantic-exploration.html">prior state-of-the-art</a>, respectively, on the <a href="https://eval.ai/web/challenges/challenge-page/580/leaderboard/1634">Habitat ObjectNav Challenge</a>. From our analysis, we propose that agents will act to simplify their visual inputs so as to smooth their RNN dynamics, and that auxiliary tasks reduce overfitting by minimizing effective RNN dimensionality; i.e. a performant ObjectNav agent that must maintain coherent plans over long horizons does so by learning smooth, low-dimensional recurrent dynamics.

## Approach
![Overview](/overview.png)
Our agent must navigate to a goal instance from RGBD-Input and a GPS-Compass sensor. Building on <a href="https://arxiv.org/abs/2007.04561">Ye et al, 2020</a>, we introduce new auxiliary tasks, a semantic segmentation as visual input, a Semantic Goal Exists feature which describes the fraction of the frame occupied by the goal class, and a method for tethering secondary policies that learns from its own reward signal with off-policy updates. We encourage the acting policy to explore and the tethered policy to perform efficient ObjectNav.

## Qualitative Examples
On this page, we provide selected videos of various agents on the validation split, highlighting different failure modes as well as agent instability. The videos show, from left to right, RGB, Depth, Semantic Input (either ground truth or RedNet segmentations), and the top-down map (the agent does not receive a top-down map). The top down map shows goals as red squares and valid success zones in pink. The annotations at the top left of each video show the coverage and the SGE fed into the agent. Unless otherwise indicated, the videos use GT segmentation.

### Selected Successes from 4-Action, 6-Action, and 6-Action + Tether
All of the following have a **cushion** goal.
*4-Action: SPL 0.66*.
<video width="960" height="180" controls>
  <source src="videos/base4_gt_short_0.66.mp4" type="video/mp4">
</video>
*6-Action: SPL 0.57*.
<video width="960" height="180" controls>
  <source src="videos/base_gt_short_0.57.mp4" type="video/mp4">
</video>
*6-Action + Tether: SPL 0.82*.
<video width="960" height="180" controls>
  <source src="videos/tether_gt_short_0.82.mp4" type="video/mp4">
</video>
*4-Action: SPL 0.08*.
<video width="960" height="180" controls>
  <source src="videos/base4_gt_long_0.08.mp4" type="video/mp4">
</video>
*6-Action: SPL 0.06*
<video width="960" height="180" controls>
  <source src="videos/base_gt_long_0.06.mp4" type="video/mp4">
</video>
*6-Action + Tether: SPL 0.60*
<video width="960" height="180" controls>
  <source src="videos/tether_gt_long_0.60.mp4" type="video/mp4">
</video>

### Failure Modes
We present samples of failure modes noted in the behavioral study for the base agent.

*Plateau (Spawn)*
<video width="960" height="180" controls>
  <source src="videos/base_gt_debris_spawn.mp4" type="video/mp4">
</video>
*Plateau*
<video width="960" height="180" controls>
  <source src="videos/base_gt_debris.mp4" type="video/mp4">
</video>
*Last Mile*
<video width="960" height="180" controls>
  <source src="videos/base_gt_lastmile.mp4" type="video/mp4">
</video>
*Loop*
<video width="960" height="180" controls>
  <source src="videos/base_gt_loop.mp4" type="video/mp4">
</video>
*Open*
<video width="960" height="180" controls>
  <source src="videos/base_gt_open.mp4" type="video/mp4">
</video>
*Dataset Bug*: Bed (Goal) is in segmentation
<video width="960" height="180" controls>
  <source src="videos/base_gt_goal_bug.mp4" type="video/mp4">
</video>
*Detection*: Agent glimpses goal at ~0:15 - 0:20.
<video width="960" height="180" controls>
  <source src="videos/base_gt_detect.mp4" type="video/mp4">
</video>
*Commitment*: Agent ignores
<video width="960" height="180" controls>
  <source src="videos/base_gt_commit.mp4" type="video/mp4">
</video>
*Explore*
<video width="960" height="180" controls>
  <source src="videos/base_gt_explore.mp4" type="video/mp4">
</video>
*Quit (Tether Failure)*
<video width="960" height="180" controls>
  <source src="videos/tether_gt_quit.mp4" type="video/mp4">
</video>

### Unstable Behavior on Zero-Shot Transfer to RedNet Segmentation
*4-Action*
<video width="960" height="180" controls>
  <source src="videos/base4_instability.mp4" type="video/mp4">
</video>
*6-Action*
<video width="960" height="180" controls>
  <source src="videos/base_instability.mp4" type="video/mp4">
</video>

*4-Action (Example 2)*
<video width="960" height="180" controls>
  <source src="videos/base4_instability2.mp4" type="video/mp4">
</video>
*6-Action (Example 2)*
<video width="960" height="180" controls>
  <source src="videos/base_instability2.mp4" type="video/mp4">
</video>


### Additional Random Successes for 6-Action Base Agent
*Goal: Chair*
<video width="960" height="180" controls>
  <source src="videos/base_gt_flavor1.mp4" type="video/mp4">
</video>
*Goal: Chest of Drawers*
<video width="960" height="180" controls>
  <source src="videos/base_gt_flavor2.mp4" type="video/mp4">
</video>
*Goal: Chair*
<video width="960" height="180" controls>
  <source src="videos/base_gt_flavor3.mp4" type="video/mp4">
</video>

### Additional 6-Action Base Episodes with RedNet Segmentation

*Goal: Plant*
<video width="960" height="180" controls>
  <source src="videos/base_rednet_flavor1.mp4" type="video/mp4">
</video>
*Goal: Picture*
<video width="960" height="180" controls>
  <source src="videos/base_rednet_flavor2.mp4" type="video/mp4">
</video>
