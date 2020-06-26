---
title:  "D&D and Reinforcement Learning"
date:   2020-06-26 15:04:23
categories: [machine-learning]
tags: [machine-learning]
header:
  image: /assets/images/dice.jpg
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
---

# Dungeons and Data

This post is a first in a series of posts of applying machine learning to dungeons and dragons. This blog post details this I've learned from applying reinforcement learning algorithms in a simple [Dungeons and Dragons combat](https://www.youtube.com/watch?v=7tnrATiclg4) scenario. Future blog posts will include:

 * Applying RL algorithms to more complicated combat scenarios
 * Applying NLP to the story telling aspect of D&D:
     * Summarization of Critical Role episodes
     * Question and answering of events within a Critical Role episode
     
The code associated with this blog post can be seen [in this github repo](https://github.com/AndrewLim1990/dungeonsNdata).

## Combat Scenario Description

This section discusses the environment in which combat takes place. 

In order to first allow for learning in a simple environment, the combat takes place in a 50ft x 50ft room and only involves two combatants:

1. Leotris:
    * Hit points: 25
    * Armor class: 16
    * Speed: 30ft
    * Shoot arrow attack: 
        * 60 ft range
        * Hit bonus: +5
        * Damage dice: 1d12
        * Damage bonus: +3
    * Initial coordinates: (5, 5)


2. Strahd:
    * Hit points: 200
    * Armor class: 16
    * Speed: 30ft
    * Vampire bite attack:
        * 5 ft range
        * Hit bonus: +10
        * Damage dice: 3d12
        * Damage bonus: +10
    * Initial coordinates: (5, 10)
    
Each combatant, along with the attacks listed above, is allowed the following actions:
    * MoveLeft: move left 5ft
    * MoveRight: move right 5ft
    * MoveDown: move down 5ft
    * MoveUp: move up 5ft
    * EndTurn: ends turn
    
Additionally, a "time limit" of 1500 actions was implemented. In other words, agents were permitted to take a maximum of 1500 actions within a single combat encounter. 

## Goals and Learning Environment

For this experiment, `Leotris` was assigned one of the RL algorithms described below while `Strahd` was assigned a `Random` strategy in which actions were chosen at random.

The scenario was purposefully set up such that Strahd had an obvious advantage. However, if Leotris is able learn to keep his distance from Strahd and use his `ShootArrow` attack, he should be able to win a considerable amount of the games.

The following are learning goals I envisioned for this project:

* `Leotris` out performs strategy of taking random actions.
* `Leotris` learns to dealt damage. The challenge with this goal is that the agent must learn that it cannot just repeatedly take the `ShootArrow` action within the same turn. Instead, due to D&D combat rules, the agent must take the `EndTurn` action in between each `ShootArrow` usage if damage is to be done.
* `Leotris` learns to avoid damage. `Leotris` can only take damage from `Strahd` if they are within 5 ft of each other. 

In order to accomplish the above goals, the agent's "state" consists of the following:

1. Current hit points
2. Enemy hit points
3. Current x-coordinate
4. Current y-coordinate
5. Enemy x-coordinate
6. Enemy y-coordinate
7. Number of attacks used this turn
8. Remaining movement available this turn
9. Time limit remaining (as a percentage of the time limit)

Additionally, a reward of `5` was given if `Leotris` was the winner. Otherwise, a reward of `0` was given if `Leotris` had reached the "time limit" (1500 actions) or his hit points were reduced to zero. 

## Results and Discussion

### Random

Below are the result of both `Strahd` and `Leotris` using actions at random.

![](/assets/images/random.png)

The above was used to evaluate whether goal 1 had been achieved.

### Dueling Double Deep Q-Learning Network

When I initially implemented a dueling double deep Q network, the follow results were achieved:

![dddqn_results](/assets/images/double_dueling_DQN_high_alpha.png)

Similar results as the above were observed for a vanilla DQN, and a double DQN. Although these results did not show evidence that a reasonable strategy had been learned, the agent began to exhibit improved performance with a couple key adjustments.

The first important adjustment was optimizing the learning rate $\alpha$. I started to see a reasonable strategy being learned if $\alpha$ was tuned correctly. The problem was that the learning rate was initially too large and the agent could not learn how to "escape" areas of suboptimal strategies within the parameter space.

![damage_reward_results](/assets/images/damage_reward_dddqn.png)

Another adjustment that had to be made was to the linear decay of the ϵ-greedy exploration strategy. The ϵ-greedy strategy was implemented such that the agent started with an ϵ-exploration probability of 90% which decayed linearly to 5% over 50,000 actions. That is to say, at the beginning stages of exploration, the agent would take the action believed to be optimal 10% of the time. The remaining 90% of the time, the agent would "explore" by taking a random action. The ϵ-value would decay linearly down to a 5% probability of taking a random action and a 95% probability of taking an action believed to be optimal. 

During training, the agent seemed to learn that the action `ShootArrow` was the best action to perform regardless of the `current_state`. Unfortunately, the agent would attempt to `ShootArrow` despite the fact that it had already used the action earlier within the same turn, which is against the rules. As a result, the action would be ignored and the agent would be prompted for its next action until the `EndTurn` action was chosen or the time limit was reached. The agent never learned when to take the `EndTurn` action consistently resulted a `Timeout` terminal state as observed in the plot above.

Here are the biggest problems/obstacles I had to contend with and the steps I took to address them:

1. **ϵ decay rate was too fast**: Perhaps the neural network did not have enough time/epochs to learn that taking the `EndTurn` action would be beneficial during higher exploration phase. I decreased the decay rate by a factor of 100. i.e. I increased the number of actions to go from 90% to 10% exploration from 50,000 actions to 5,000,000 actions. This seemed to help quite a lot.
2. **Sparse rewards**: Perhaps only providing a reward for achieving a victory was too sparse. To address this, I structured the rewards such that the agent was rewarded every time it was able to do damage. This helped a great deal with the agent even learning to alter between `ShootArrow` and `EndTurn`. However, I decided to return the reward structure back to the original as I was more interested in a sparser reward setting
3. **Catastrophic forgetting**: Looking at [this](https://ai.stackexchange.com/questions/10822/what-is-happening-when-a-reinforcement-learning-agent-trains-itself-out-of-desir) stack exchange post, the user is asking why DQNs sometimes train themselves out of desired behavior. This was observed when the rewards were altered to reward any damage being delt. One answer given suggested that this could be a result of "catastrophic forgetting" (detailed in the post). To combat this, I tried to decrease the learning rate and use prioritized experience replay. I observed the most success when I decreased the learning rate from 1e-3 to 1e-5. Another suggestion was to keep experiences from early stages of exploration within memory. I have not tried this one yet. 

Once the above were addressed, the agent was able to learn a reasonable strategy which exhibited the following results:

![dddqn_results](/assets/images/double_dueling_DQN.png)

### Proximal Policy Iteration (PPO)

A simple implementation of [PPO](https://arxiv.org/abs/1707.06347) managed also managed to learn a reasonable strategy:

![](/assets/images/PPO.png)

I believe that there were a couple contributing factors in obtaining these results. Again it was vital that the learning rate be sufficiently small ($\alpha$=1e-5). With a larger learning rate of ($\alpha$=1e-3), the following results were observed:

![](/assets/images/PPO_high_alpha.png)

In the case depicted above, the agent seems to have gotten into a parameter space in which it could not recover from

I found that the PPO agent was not as sensitive to hyper parameter tuning and worked better out of the box. I think one of the largest contributing factors is the fact that the agent did not use an epsilon greedy like exploration strategy. Intead, PPO selects actions in a more stochastic nature compared to the epsilon greedy approach once its reached low exploration states. As a result, agent was less likely to get "stuck" in a bad area.

Although I was not able achieve as good results as the dueling double DQN, I don't think it that this is indicative of the potential of PPO. I spent a lot more time adjusting hyper parameters and adding bells and whistles for DQNs. I believe of the PPO was made deeper and the learning rate was properly tuned, it would be able to match the performance of the DQN.

## Conclusion

Here are some top things I took away from this project:
1. Be patient. Reinforcement learning takes a long time.
2. Optimizing for a good learning rate is important. Learning rate is almost always the most imporant hyper parameter.
3. Implement algorithms in small and simple scenarios. This helps immensely with tracking down bugs and speeding up the speed at which you can iterate.
4. It's a good idea to make your solution fast and scalable. This is an area I neglected and the main source of frustration for me. Operating on a slow iteration cycle was painful with instances where I would be waiting days for the agent to learn a reasonable strategy only to find out there was a bug somewhere or that I wanted to adjust a hyperparameter. If I had made my solution more scalable, I could have cut down on the time waiting around.
5. Don't let perfection get in the way of progress. Is my code a piece of crap? Yes. Did I learn a lot by doing this? Yes x 100. While building, I found it hard to resist the temptation to backtrack in order tooptimize/refactor large portions of code. Rather than get bogged down by this, I opted to push onward just to get a functional solution. In hindsight, I am very glad I opted to do this because there were many other more important and interesting (relevant to RL) problems that arose. 
