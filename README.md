On this page, we provide selected videos of various agents on the validation split, highlighting different failure modes as well as agent instability. The videos show, from left to right, RGB, Depth, Semantic Input (either ground truth or RedNet segmentations), and the top-down map (the agent does not receive a top-down map). The top down map shows goals as red squares and valid success zones in pink. The annotations at the top left of each video show the coverage and the SGE fed into the agent.

## Selected Successes from 4-Action, 6-Action, and 6-Action + Tether
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

## Failure Modes
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

## Unstable Behavior on Zero-Shot Transfer to RedNet Segmentation
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


## Additional Random Successes for 6-Action Base Agent
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
