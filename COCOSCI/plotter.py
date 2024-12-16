import matplotlib.pyplot as plt
import numpy as np

''' Trajectory one ''' 
# Human data - use 1 for continue and 0 for slow down
traj1_step1_a_all = [0.45, 0.25, 0.35, 0.4, 0.25]
traj1_step1_b_all = [0.1, 0.25, 0.15, 0.2, 0.25 ]
traj1_step1_c_all = [0.45, 0.5, 0.5, 0.4, 0.5]
traj1_step1_decisions_all = [1, 1, 1, 1, 1]
traj1_step1_a_avg = sum(traj1_step1_a_all)/ len(traj1_step1_a_all)
traj1_step1_b_avg = sum(traj1_step1_b_all) / len(traj1_step1_b_all)
traj1_step1_c_avg = sum(traj1_step1_c_all) / len(traj1_step1_c_all) 
traj1_step1_decisions_avg = sum(traj1_step1_decisions_all) / len(traj1_step1_decisions_all)  

traj1_step2_a_all = [0.45, .4, .4, .4, .3]
traj1_step2_b_all = [0.1, .2, .2, .2, .3]
traj1_step2_c_all = [0.45, .4, .4, .4, .4]
traj1_step2_decisions_all = [1, 0, 1, 1, 1]
traj1_step2_a_avg = sum(traj1_step2_a_all) / len(traj1_step2_a_all)  # Average of traj1_step2_a_all
traj1_step2_b_avg = sum(traj1_step2_b_all) / len(traj1_step2_b_all)  # Average of traj1_step2_b_all
traj1_step2_c_avg = sum(traj1_step2_c_all) / len(traj1_step2_c_all)
traj1_step2_decisions_avg = sum(traj1_step2_decisions_all) / len(traj1_step2_decisions_all) 

traj1_step3_a_all = [0.45, 0.5, .7, .5, .5]
traj1_step3_b_all = [0.45, 0.5, .25, .25, 0]
traj1_step3_c_all = [0.1, 0, .05, .25, .5]
traj1_step3_decisions_all = [1, 1, 1, 1, 1]
traj1_step3_a_avg = sum(traj1_step3_a_all) / len(traj1_step3_a_all)  # Average of traj1_step3_a_all
traj1_step3_b_avg = sum(traj1_step3_b_all) / len(traj1_step3_b_all)  # Average of traj1_step3_b_all
traj1_step3_c_avg = sum(traj1_step3_c_all) / len(traj1_step3_c_all)
traj1_step3_decisions_avg = sum(traj1_step3_decisions_all) / len(traj1_step3_decisions_all) 


traj1_step4_a_all = [.475, 0.75, .75, .45, .6]
traj1_step4_b_all = [.475, 0.25, .25, .35, .1]
traj1_step4_c_all = [.05, 0, 0, .2, .3]
traj1_step4_decisions_all = [0, 0, 0, 0, 0]
traj1_step4_a_avg = sum(traj1_step4_a_all) / len(traj1_step4_a_all)  # Average of traj1_step4_a_all
traj1_step4_b_avg = sum(traj1_step4_b_all) / len(traj1_step4_b_all)  # Average of traj1_step4_b_all
traj1_step4_c_avg = sum(traj1_step4_c_all) / len(traj1_step4_c_all) 
traj1_step4_decisions_avg = sum(traj1_step4_decisions_all) / len(traj1_step4_decisions_all) 

# Model 1
traj1_step4_a_model1 = 0.6652409557748218
traj1_step4_b_model1 = 0.2447284710547976
traj1_step4_c_model1 = 0.09003057317038045
traj1_step4_decisions_model1 = 0
traj1_step4_a_model1_prog = 0.7306791292026887
traj1_step4_b_model1_prog = 0.20934307548219697
traj1_step4_c_model1_prog = 0.05997779531511428
traj1_step4_decisions_model1_prog = 0

# Model 2
traj1_step1_a_model2 = 1/3
traj1_step1_b_model2 = 1/3
traj1_step1_c_model2 = 1/3
traj1_step1_decisions_model2 = 1
traj1_step2_a_model2 = (0.7458838474594635 + 0.4607560991023355 + 1/3 + 0.4607560991023355)/4
traj1_step2_b_model2 = (0.1270580762702683 + 0.4607560991023355 + 1/3 + 0.4607560991023355)/4
traj1_step2_c_model2 = (0.1270580762702683 + 0.07848780179532909 + 1/3 + 0.07848780179532909)/4

traj1_step2_decisions_model2 = (0 + 0 + 1 + 0)/4
traj1_step3_a_model2 = (0.8195669502375433 + 0.8195669502375433 + 0.7828471494848475 +  0.527522003277725)/4
traj1_step3_b_model2 = (0.06676379801311666 + 0.06676379801311666 + 0.10857642525757628 + 0.4295048479586496)/4
traj1_step3_c_model2 = (0.11366925174934003 + 0.11366925174934003 + 0.10857642525757628 + 0.04297314876362537)/4

traj1_step3_decisions_model2 = 0
traj1_step4_a_model2 = (0.46869926720391 + 0.9768432517889117 + 0.5286522462378767 + 0.8718945240359337)/4
traj1_step4_b_model2 = (0.46869926720391 + 0.0009315225008765456 + 0.3504491929624257 + 0.011651538726098126)/4
traj1_step4_c_model2 = (0.06260146559218006 + 0.022225225710211773 + 0.12089856079969762 +0.116453937237968)/4
traj1_step4_decisions_model2 = 0

''' Trajectory two ''' 
# Human data
traj2_step1_a_all = [.25, 1/3, .25, .4, .4]
traj2_step1_b_all = [.25, 1/3, .25, .2, .2]
traj2_step1_c_all = [.5, 1/3, .5, .4, .4]
traj2_step1_decisions_all = [0, 0, 0, 0, 0]
traj2_step1_a_avg = sum(traj2_step1_a_all) / len(traj2_step1_a_all)
traj2_step1_b_avg = sum(traj2_step1_b_all) / len(traj2_step1_b_all)
traj2_step1_c_avg = sum(traj2_step1_c_all) / len(traj2_step1_c_all)
traj2_step1_decisions_avg = sum(traj2_step1_decisions_all) / len(traj2_step1_decisions_all) 

traj2_step2_a_all = [.4, 0.5, 0.75, .4, .55]
traj2_step2_b_all = [.4, 0.4, .25, .3, .15]
traj2_step2_c_all = [.2, .1, 0, .3, .3]
traj2_step2_decisions_all = [0, 0, 1, 1, 1]
traj2_step2_a_avg = sum(traj2_step2_a_all) / len(traj2_step2_a_all)
traj2_step2_b_avg = sum(traj2_step2_b_all) / len(traj2_step2_b_all)
traj2_step2_c_avg = sum(traj2_step2_c_all) / len(traj2_step2_c_all)
traj2_step2_decisions_avg = sum(traj2_step2_decisions_all) / len(traj2_step2_decisions_all) 

traj2_step3_a_all = [.5, 0.75, 0.75, .5, .5]
traj2_step3_b_all = [.4, .25, 0.25, .35, .4]
traj2_step3_c_all = [.1, 0, 0, .15, .1]
traj2_step3_decisions_all = [0, 0, 1, 0, 0]
traj2_step3_a_avg = sum(traj2_step3_a_all) / len(traj2_step3_a_all)
traj2_step3_b_avg = sum(traj2_step3_b_all) / len(traj2_step3_b_all)
traj2_step3_c_avg = sum(traj2_step3_c_all) / len(traj2_step3_c_all)
traj2_step3_decisions_avg = sum(traj2_step3_decisions_all) / len(traj2_step3_decisions_all) 


# Model 1
traj2_step4_a_model1 = 0.7869860421615985
traj2_step4_b_model1 = 0.10650697891920076
traj2_step4_c_model1 = 0.10650697891920076
traj2_step4_decisions_model1 = 0
traj2_step4_a_model1_prog = 0.8779886385221453
traj2_step4_b_model1_prog = 0.061005680738927405
traj2_step4_c_model1_prog = 0.061005680738927405
traj2_step4_decisions_model1_prog = 0

# Model 2
traj2_step1_a_model2 = 1/3
traj2_step1_b_model2 = 1/3
traj2_step1_c_model2 = 1/3
traj2_step1_decisions_model2 = 1
traj2_step2_a_model2 = 0.5930958654456907
traj2_step2_b_model2 = 0.20345206727715465
traj2_step2_c_model2 = 0.20345206727715465
traj2_step2_decisions_model2 = 0
traj2_step3_a_model2 = 0.8988392647585066
traj2_step3_b_model2 = 0.050580367620746663
traj2_step3_c_model2 = 0.050580367620746663
traj2_step3_decisions_model2 = 0

''' Trajectory 3 '''
# Human data
traj3_step1_a_all = [.05, .15, 0, .2, .15]
traj3_step1_b_all = [.9, .75, .75, .65, .75]
traj3_step1_c_all = [.05, .1, .25, .15, .1]
traj3_step1_decisions_all = [0, 1, 1, 0, 0]
traj3_step1_a_avg = sum(traj3_step1_a_all) / len(traj3_step1_a_all)
traj3_step1_b_avg = sum(traj3_step1_b_all) / len(traj3_step1_b_all)
traj3_step1_c_avg = sum(traj3_step1_c_all) / len(traj3_step1_c_all)
traj3_step1_decisions_avg = sum(traj3_step1_decisions_all) / len(traj3_step1_decisions_all) 

traj3_step2_a_all = [.2, 0, 0, 1/3, .2]
traj3_step2_b_all = [.3, 0.25, .25, 1/3, .4]
traj3_step2_c_all = [.5, .75, .75, 1/3, .4]
traj3_step2_decisions_all = [0, 1, 1, 1, 1]
traj3_step2_a_avg = sum(traj3_step2_a_all) / len(traj3_step2_a_all)
traj3_step2_b_avg = sum(traj3_step2_b_all) / len(traj3_step2_b_all)
traj3_step2_c_avg = sum(traj3_step2_c_all) / len(traj3_step2_c_all)
traj3_step2_decisions_avg = sum(traj3_step2_decisions_all) / len(traj3_step2_decisions_all)
print(traj3_step2_a_avg, traj3_step2_b_avg, traj3_step2_c_avg)


# Model 1
traj3_step4_a_model1 = 0.21194155761708544
traj3_step4_b_model1 = 0.21194155761708544
traj3_step4_c_model1 = 0.5761168847658291
traj3_step4_decisions_model1 = 0
traj3_step4_a_model1_prog = 0.15428077298188622
traj3_step4_b_model1_prog = 0.15428077298188622
traj3_step4_c_model1_prog = 0.6914384540362276
traj3_step4_decisions_model1_prog = 0

# Model 2
traj3_step1_a_model2 = 1/3
traj3_step1_b_model2 = 1/3
traj3_step1_c_model2 = 1/3
traj3_step1_decisions_model2 = 1
traj3_step2_a_model2 = 0.1270580762702683
traj3_step2_b_model2 = 0.1270580762702683 
traj3_step2_c_model2 = 0.7458838474594635
traj3_step2_decisions_model2 = 0

''' Plot Goal Likelihoods'''
color1 = "#8C7A6B"
color2 = "#5C6B6B"
color3 = "#A3B8A1"
color4 = "#D1B7A1"

# Trajectory 1 (Data from previous example)
human_data_a_1 = [traj1_step1_a_avg, traj1_step2_a_avg, traj1_step3_a_avg, traj1_step4_a_avg]
model1_data_a_1 = [1/3, 1/3, 0.5761, traj1_step4_a_model1]
model2_data_a_1 = [traj1_step1_a_model2, traj1_step2_a_model2, traj1_step3_a_model2, traj1_step4_a_model2]
model2_prog_a_1 = [1/3, 1/3, 0.6548, traj1_step4_a_model1_prog]

# Goal b and c for traj1 (similar to traj1_goal_a)
human_data_b_1 = [traj1_step1_b_avg, traj1_step2_b_avg, traj1_step3_b_avg, traj1_step4_b_avg]
model1_data_b_1 = [1/3, 1/3, .2119, traj1_step4_b_model1]
model2_data_b_1 = [traj1_step1_b_model2, traj1_step2_b_model2, traj1_step3_b_model2, traj1_step4_b_model2]
model2_prog_b_1 = [1/3, 1/3, .1726, traj1_step4_b_model1_prog]

human_data_c_1 = [traj1_step1_c_avg, traj1_step2_c_avg, traj1_step3_c_avg, traj1_step4_c_avg]
model1_data_c_1 = [1/3, 1/3, .2119, traj1_step4_c_model1]
model2_data_c_1 = [traj1_step1_c_model2, traj1_step2_c_model2, traj1_step3_c_model2, traj1_step4_c_model2]
model2_prog_c_1 = [1/3, 1/3, .1726, traj1_step4_c_model1_prog]

# Trajectory 2 (Data for traj2)
human_data_a_2 = [traj2_step1_a_avg, traj2_step2_a_avg, traj2_step3_a_avg]
model1_data_a_2 = [1/3, 0.5761, traj2_step4_a_model1]
model2_data_a_2 = [traj2_step1_a_model2, traj2_step2_a_model2, traj2_step3_a_model2]
model2_prog_a_2 = [1/3, 0.6914, traj2_step4_a_model1_prog]

human_data_b_2 = [traj2_step1_b_avg, traj2_step2_b_avg, traj2_step3_b_avg]
model1_data_b_2 = [1/3, 0.2119, traj2_step4_b_model1]
model2_data_b_2 = [traj2_step1_b_model2, traj2_step2_b_model2, traj2_step3_b_model2]
model2_prog_b_2 = [1/3, 0.1543, traj2_step4_b_model1_prog]

human_data_c_2 = [traj2_step1_c_avg, traj2_step2_c_avg, traj2_step3_c_avg]
model1_data_c_2 = [1/3, .2119, traj2_step4_c_model1]
model2_data_c_2 = [traj2_step1_c_model2, traj2_step2_c_model2, traj2_step3_c_model2]
model2_prog_c_2 = [1/3, 0.1543, traj2_step4_c_model1_prog]

# Trajectory 3 (Data for traj3)
human_data_a_3 = [traj3_step1_a_avg, traj3_step2_a_avg]
model1_data_a_3 = [1/3, traj3_step4_a_model1]
model2_data_a_3 = [traj3_step1_a_model2, traj3_step2_a_model2]
model2_prog_a_3 = [1/3, traj3_step4_a_model1_prog]

human_data_b_3 = [traj3_step1_b_avg, traj3_step2_b_avg]
model1_data_b_3 = [1/3, traj3_step4_b_model1]
model2_data_b_3 = [traj3_step1_b_model2, traj3_step2_b_model2]
model2_prog_b_3 = [1/3, traj3_step4_b_model1_prog]

human_data_c_3 = [traj3_step1_c_avg, traj3_step2_c_avg]
model1_data_c_3 = [1/3, traj3_step4_c_model1]
model2_data_c_3 = [traj3_step1_c_model2, traj3_step2_c_model2]
model2_prog_c_3 = [1/3, traj3_step4_c_model1_prog]

# Step labels
steps = ['Step 1', 'Step 2', 'Step 3', 'Step 4']
x = np.arange(len(steps))  # X-axis positions for the steps
width = 0.15  # Width of each bar

# Trajectory 1 Plot
plt.figure(figsize=(12, 6))

# Goal A
plt.subplot(1, 3, 1)
x_labels = ['Step 1', 'Step 2', 'Step 3', 'Step 4']
bar_width = 0.2
index = np.arange(len(x_labels))

plt.bar(index, human_data_a_1, bar_width, label='Human', color=color1)
plt.bar(index + bar_width, model1_data_a_1, bar_width, label='Direct Likelihood', color=color2)
plt.bar(index + 2*bar_width, model2_prog_a_1, bar_width, label='Direct Likelihood w/ Progress', color=color4)
plt.bar(index + 3*bar_width, model2_data_a_1, bar_width, label='Policy Learning', color=color3)

plt.title('Trajectory 1 - Goal = Arena')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.xticks(index + 1.5 * bar_width, x_labels)
plt.ylim(0, 1)
plt.legend()

# Goal B
plt.subplot(1, 3, 2)
plt.bar(index, human_data_b_1, bar_width, label='Human', color=color1)
plt.bar(index + bar_width, model1_data_b_1, bar_width, label='Direct Likelihood', color=color2)
plt.bar(index + 2*bar_width, model2_prog_b_1, bar_width, label='Direct Likelihood w/ Progress', color=color4)
plt.bar(index + 3*bar_width, model2_data_b_1, bar_width, label='Policy Learning', color=color3)

plt.title('Trajectory 1 - Goal = Bank')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.xticks(index + 1.5 * bar_width, x_labels)
plt.ylim(0, 1)
plt.legend()

# Goal C
plt.subplot(1, 3, 3)
plt.bar(index, human_data_c_1, bar_width, label='Human', color=color1)
plt.bar(index + bar_width, model1_data_c_1, bar_width, label='Direct Likelihood', color=color2)
plt.bar(index + 2*bar_width, model2_prog_c_1, bar_width, label='Direct Likelihood w/ Progress', color=color4)
plt.bar(index + 3*bar_width, model2_data_c_1, bar_width, label='Policy Learning', color=color3)

plt.title('Trajectory 1 - Goal = Cafe')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.xticks(index + 1.5 * bar_width, x_labels)
plt.ylim(0, 1)
plt.legend()

# Show Trajectory 1 Plot
plt.tight_layout()
plt.show()

# ------------------------------------

# Trajectory 2 Plot
plt.figure(figsize=(12, 6))

# Goal A
plt.subplot(1, 3, 1)
x_labels = ['Step 1', 'Step 2', 'Step 3']
bar_width = 0.2
index = np.arange(len(x_labels))

plt.bar(index, human_data_a_2, bar_width, label='Human', color=color1)
plt.bar(index + bar_width, model1_data_a_2, bar_width, label='Direct Likelihood', color=color2)
plt.bar(index + 2*bar_width, model2_prog_a_2, bar_width, label='Direct Likelihood w/ Progress', color=color4)
plt.bar(index + 3*bar_width, model2_data_a_2, bar_width, label='Policy Learning', color=color3)

plt.title('Trajectory 2 - Goal = Arena')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.xticks(index + 1.5 * bar_width, x_labels)
plt.ylim(0, 1)
plt.legend()

# Goal B
plt.subplot(1, 3, 2)
plt.bar(index, human_data_b_2, bar_width, label='Human', color=color1)
plt.bar(index + bar_width, model1_data_b_2, bar_width, label='Direct Likelihood', color=color2)
plt.bar(index + 2*bar_width, model2_prog_b_2, bar_width, label='Direct Likelihood w/ Progress', color=color4)
plt.bar(index + 3*bar_width, model2_data_b_2, bar_width, label='Policy Learning', color=color3)

plt.title('Trajectory 2 - Goal = Bank')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.xticks(index + 1.5 * bar_width, x_labels)
plt.ylim(0, 1)
plt.legend()

# Goal C
plt.subplot(1, 3, 3)
plt.bar(index, human_data_c_2, bar_width, label='Human', color=color1)
plt.bar(index + bar_width, model1_data_c_2, bar_width, label='Direct Likelihood', color=color2)
plt.bar(index + 2*bar_width, model2_prog_c_2, bar_width, label='Direct Likelihood w/ Progress', color=color4)
plt.bar(index + 3*bar_width, model2_data_c_2, bar_width, label='Policy Learning', color=color3)

plt.title('Trajectory 2 - Goal = Cafe')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.xticks(index + 1.5 * bar_width, x_labels)
plt.ylim(0, 1)
plt.legend()

# Show Trajectory 2 Plot
plt.tight_layout()
plt.show()

# ------------------------------------

# Trajectory 3 Plot
plt.figure(figsize=(12, 6))

# Goal A
plt.subplot(1, 3, 1)
x_labels = ['Step 1', 'Step 2']
bar_width = 0.2
index = np.arange(len(x_labels))

plt.bar(index, human_data_a_3, bar_width, label='Human', color=color1)
plt.bar(index + bar_width, model1_data_a_3, bar_width, label='Direct Likelihood', color=color2)
plt.bar(index + 2*bar_width, model2_prog_a_3, bar_width, label='Direct Likelihood w/ Progress', color=color4)
plt.bar(index + 3*bar_width, model2_data_a_3, bar_width, label='Policy Learning', color=color3)

plt.title('Trajectory 3 - Goal = Arena')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.xticks(index + 1.5 * bar_width, x_labels)
plt.ylim(0, 1)
plt.legend()

# Goal B
plt.subplot(1, 3, 2)
plt.bar(index, human_data_b_3, bar_width, label='Human', color=color1)
plt.bar(index + bar_width, model1_data_b_3, bar_width, label='Direct Likelihood', color=color2)
plt.bar(index + 2*bar_width, model2_prog_b_3, bar_width, label='Direct Likelihood w/ Progress', color=color4)
plt.bar(index + 3*bar_width, model2_data_b_3, bar_width, label='Policy Learning', color=color3)

plt.title('Trajectory 3 - Goal = Bank')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.xticks(index + 1.5 * bar_width, x_labels)
plt.ylim(0, 1)
plt.legend()

# Goal C
plt.subplot(1, 3, 3)
plt.bar(index, human_data_c_3, bar_width, label='Human', color=color1)
plt.bar(index + bar_width, model1_data_c_3, bar_width, label='Direct Likelihood', color=color2)
plt.bar(index + 2*bar_width, model2_prog_c_3, bar_width, label='Direct Likelihood w/ Progress', color=color4)
plt.bar(index + 3*bar_width, model2_data_c_3, bar_width, label='Policy Learning', color=color3)

plt.title('Trajectory 3 - Goal = Cafe')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.xticks(index + 1.5 * bar_width, x_labels)
plt.ylim(0, 1)
plt.legend()

# Show Trajectory 3 Plot
plt.tight_layout()
plt.show()


''' Plot Decisions '''
# Trajectory 1 
steps = ['Step 1', 'Step 2', 'Step 3', 'Step 4']
methods = ['Human', 'Direct Likelihood', 'Direct Likelihood w/ Progress', 'Policy Learning']
colors = ["#8C7A6B", "#5C6B6B", "#D1B7A1", "#A3B8A1"]

continue_traj1 = np.array([[traj1_step1_decisions_avg, traj1_step2_decisions_avg, traj1_step3_decisions_avg, traj1_step4_decisions_avg],
                           [1, 1, 1, traj1_step4_decisions_model1],
                           [1, 1, 1, traj1_step4_decisions_model1_prog],
                           [traj1_step1_decisions_model2, traj1_step2_decisions_model2, traj1_step3_decisions_model2, traj1_step4_decisions_model2]])
                            
slow_traj1 = 1 - continue_traj1

# Create a plot for all the steps
fig, ax = plt.subplots(figsize=(12, 8))

# Bar width and positions
bar_width = 0.15
index = np.arange(len(steps))

# Loop over the methods and plot stacked bars for each method at each step
for i, method in enumerate(methods):
    bars_slow = ax.barh(index + i * bar_width, slow_traj1[i], bar_width, color='#2F1B12', left=continue_traj1[i], label='Slow Down')
    bars_continue = ax.barh(index + i * bar_width, continue_traj1[i], bar_width, color=colors[i], label=f'{method} - Continue')
    
    for j, bar in enumerate(bars_slow):
        total_width = continue_traj1[i][j] + slow_traj1[i][j]  # Total width of the stacked bar
        # Add the method label on the rightmost edge
        height = bar.get_height()
        ax.text(total_width - 0.01, bar.get_y() + bar.get_height() / 2, method, ha='right', va='center', fontsize=15, color='white')

    

# Set labels and title
ax.set_yticks(index + bar_width * 1.5)  # Place the step names at the center of each group
ax.set_yticklabels(steps)
ax.set_xlabel('Decision')
ax.set_title('Trajectory 1 Decisions with Steps for All Methods')


handles, labels = ax.get_legend_handles_labels()
unique_labels = []
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

ax.legend(unique_handles, unique_labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5)


# Show the plot
plt.tight_layout()
plt.show()

# ----------------- #

# Trajectory 2
steps = ['Step 1', 'Step 2', 'Step 3']
methods = ['Human', 'Direct Likelihood', 'Direct Likelihood w/ Progress', 'Policy Learning']
colors = ["#8C7A6B", "#5C6B6B", "#D1B7A1", "#A3B8A1"]

continue_traj1 = np.array([[traj2_step1_decisions_avg, traj2_step2_decisions_avg, traj2_step3_decisions_avg],
                           [1, 1, traj2_step4_decisions_model1],
                           [1, 1, traj2_step4_decisions_model1_prog],
                           [traj2_step1_decisions_model2, traj2_step2_decisions_model2, traj2_step3_decisions_model2]])
                            
slow_traj1 = 1 - continue_traj1

# Create a plot for all the steps
fig, ax = plt.subplots(figsize=(12, 8))

# Bar width and positions
bar_width = 0.15
index = np.arange(len(steps))

# Loop over the methods and plot stacked bars for each method at each step
for i, method in enumerate(methods):
    bars_slow = ax.barh(index + i * bar_width, slow_traj1[i], bar_width, color='#2F1B12', left=continue_traj1[i], label='Slow Down')
    bars_continue = ax.barh(index + i * bar_width, continue_traj1[i], bar_width, color=colors[i], label=f'{method} - Continue')
    
    for j, bar in enumerate(bars_slow):
        total_width = continue_traj1[i][j] + slow_traj1[i][j]  # Total width of the stacked bar
        # Add the method label on the rightmost edge
        height = bar.get_height()
        ax.text(total_width - 0.01, bar.get_y() + bar.get_height() / 2, method, ha='right', va='center', fontsize=15, color='white')

    

# Set labels and title
ax.set_yticks(index + bar_width * 1.5)  # Place the step names at the center of each group
ax.set_yticklabels(steps)
ax.set_xlabel('Decision')
ax.set_title('Trajectory 2 Decisions with Steps for All Methods')


handles, labels = ax.get_legend_handles_labels()
unique_labels = []
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

ax.legend(unique_handles, unique_labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5)


# Show the plot
plt.tight_layout()
plt.show()

# -------------------------- #
# Trajectory 3 
steps = ['Step 1', 'Step 2']
methods = ['Human', 'Direct Likelihood', 'Direct Likelihood w/ Progress', 'Policy Learning']
colors = ["#8C7A6B", "#5C6B6B", "#D1B7A1", "#A3B8A1"]

continue_traj1 = np.array([[traj3_step1_decisions_avg, traj3_step2_decisions_avg],
                           [1, traj3_step4_decisions_model1],
                           [1, traj3_step4_decisions_model1_prog],
                           [traj3_step1_decisions_model2, traj3_step2_decisions_model2]])
                            
slow_traj1 = 1 - continue_traj1

# Create a plot for all the steps
fig, ax = plt.subplots(figsize=(12, 8))

# Bar width and positions
bar_width = 0.15
index = np.arange(len(steps))

# Loop over the methods and plot stacked bars for each method at each step
for i, method in enumerate(methods):
    bars_slow = ax.barh(index + i * bar_width, slow_traj1[i], bar_width, color='#2F1B12', left=continue_traj1[i], label='Slow Down')
    bars_continue = ax.barh(index + i * bar_width, continue_traj1[i], bar_width, color=colors[i], label=f'{method} - Continue')
    
    for j, bar in enumerate(bars_slow):
        total_width = continue_traj1[i][j] + slow_traj1[i][j]  # Total width of the stacked bar
        # Add the method label on the rightmost edge
        height = bar.get_height()
        ax.text(total_width - 0.01, bar.get_y() + bar.get_height() / 2, method, ha='right', va='center', fontsize=15, color='white')

    

# Set labels and title
ax.set_yticks(index + bar_width * 1.5)  # Place the step names at the center of each group
ax.set_yticklabels(steps)
ax.set_xlabel('Decision')
ax.set_title('Trajectory 3 Decisions with Steps for All Methods')


handles, labels = ax.get_legend_handles_labels()
unique_labels = []
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

ax.legend(unique_handles, unique_labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5)

# Show the plot
plt.tight_layout()
plt.show()