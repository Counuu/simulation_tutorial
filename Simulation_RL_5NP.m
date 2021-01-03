%% Simulation of a Reinforcement Learning Model (5NP)
% This code simulates a 5-parameter RL Model, as described in 
% Guitart-Masip et al. (2012)
% Computational Psychiatry Seminar WS 20/21 Dr. Nils Kroemer
% Code written: Kirsti, Sophie, Corinna, Xin

clear all

%% Initialise paramters 
% sim_par is a structure containing all settings for the simulation

<<<<<<< Updated upstream
% Experiment settings
sim_par.n_trials = 60; %number of trials (total)
=======
%RL parameters
sim_par.alpha = 0.1;    %learning rate of the simulated agent
sim_par.beta  = 2;      %lapse of the simulated agent
sim_par.gamma  = 1;     %reward sensitivity of the simulated agent
sim_par.delta  = 1;     %punishment sensitivty of the simulated agent
%sim_par.epsilon  = 0;   %approach-avoidance bias of the simulated agent
sim_par.zeta  = 1;      %action bias of the simulated agent

%Experiment settings
sim_par.n_trials = 60; %number of trials (per condition)
>>>>>>> Stashed changes
sim_par.n_part = 1;   %number of simulated participants
sim_par.n_cond = 4; %number of conditions 
sim_par.prob = 0.8; %probability to reinforce (vs. nothing)
sim_par.reward = 1; %valence of reward
sim_par.punish = -1; %valence of punishment 

%% Simulation of Agents Behavior using Reinforcement Learning 

% RL parameters: later inspect different model paramter outcomes! 
sim_par.alpha = 0.1;    %learning rate of the simulated agent
sim_par.xi  = 0.5;      %lapse of the simulated agent
sim_par.gamma  = 1;     %reward sensitivity of the simulated agent
sim_par.delta  = 1;     %punishment sensitivty of the simulated agent
%sim_par.epsilon  = 0;   %approach-avoidance bias of the simulated agent
sim_par.zeta  = 0.2;      %action bias of the simulated agent

% Preallocation of variables to increase loop speed 
% (..)

for i = 1:sim_par.n_part %For now one sim, later with different parameter settings
    
    % ==== Stimulus conditions ============================================
    % for each experimental run stimulus conditions are
    % pseudorandomly determind (equally often, but randomly shuffled)
    each_cond = sim_par.n_trials/sim_par.n_cond; 
    stim_pres = [ones(each_cond,1);2*ones(each_cond,1);3*ones(each_cond,1);4*ones(each_cond,1)]; 
    stim_pres = stim_pres(randperm(length(stim_pres))); 
    
<<<<<<< Updated upstream
    % ==== Learning for each trial ========================================
    for t = 1:sim_par.n_trials
=======
    % Go through all stimulus conditions
    for t = 1:sim_par.n_cond
        
        % Go through all trials of one condition 
        for s = 1:sim_par.n_trials
>>>>>>> Stashed changes
        
        % First trial with initial settings (no previous knowledge)
        if t == 1 
           Q(t,s) = 0; %initial Q-value is set to zero 
           
           % On the first trial the Action Probability is random,
           % initialise other variables for storage 
           ActionWeight_go(t,1) = Q(t);
           ActionWeight_nogo(t,1) = Q(t); 
           ActionProb(t,1) = 0.5;
           a(t,1) = 0;
           
           % Generate Action (binary: go vs. no-go) from Probability 
           % > binord generates random numbers from binomial distribution (nr trials, prob of success for each trial)
           ActionChoice(t,1) = binornd(1, ActionProb(t,1));       
        
        % All subsequent trials
        else
           
            % Q-update: Rescorla-Wagner update       
            % > separate parameter for sensitivity for reward (gamma) and punishment (delta) 
            % > reinforcement t-1, as they refer to the previous trial 
            if reinforcement(t-1,1) == 1 %rewarded
               Q(t) = Q(t-1) +  sim_par.alpha* ((sim_par.gamma * reinforcement(t-1,1)) - Q(t-1)); 
            elseif reinforcement(t-1,1) == -1 %punished
               Q(t) = Q(t-1) +  sim_par.alpha* ((sim_par.delta * reinforcement(t-1,1)) - Q(t-1));  
            elseif reinforcement(t-1,1) == 0 %nothing
               Q(t) = Q(t-1) +  sim_par.alpha* ( - Q(t-1));
            end 
            
            % Action Weight for go and no-go 
            ActionWeight_go(t,1) = Q(t) + sim_par.zeta ;
            ActionWeight_nogo(t,1) =  Q(t) ;

            % Action Probability for Go (softmax function)
            n(t,1:2) = [ActionWeight_go(t,1); ActionWeight_nogo(t,1)]; 
            a = softmax(n); 
            %subplot(2,1,1), bar(n), ylabel('n')
            %subplot(2,1,2), bar(a), ylabel('a')
            
            % Action Probability for Go Action 
            ActionProb(t,1) = a(t,1) * (1 - sim_par.xi) + (sim_par.xi/2);
            
            % Action Choice Go
            ActionChoice(t,1) = binornd(1, ActionProb(t,1));   
       
        end 
        
        % Reinforcement Value Calculation: depending on Action & Stimulus 
        % Reinforcement values: +1 for reward, 0 for nothing and -1 for punishment 
        
        if ActionChoice(t,1) == 0 % No-go action 
           if stim_pres(t) == 1 %Go-to-avoid (GA)
              reinforcement(t,1) = sim_par.punish *binornd(1, sim_par.prob); 
           elseif stim_pres(t) == 2 %Go-to-win (GW) 
              reinforcement(t,1) = sim_par.reward*binornd(1, 1-sim_par.prob); 
           elseif stim_pres(t) == 3 %No-go-to-avoid (NGA)
              reinforcement(t,1) = sim_par.punish *binornd(1, 1-sim_par.prob); 
           elseif stim_pres(t) == 4 %No-go-win (NGW)
              reinforcement(t,1) = sim_par.reward*binornd(1, sim_par.prob);
           end 
  
        elseif ActionChoice(t,1) == 1 %1: Go action
           if stim_pres(t) == 1 %Go-to-avoid (GA)
              reinforcement(t,1) = sim_par.punish *binornd(1, 1-sim_par.prob); 
           elseif stim_pres(t) == 2 %Go-to-win (GW) 
              reinforcement(t,1) = sim_par.reward*binornd(1, sim_par.prob); 
           elseif stim_pres(t) == 3 %No-go-to-avoid (NGA)
              reinforcement(t,1) = sim_par.punish *binornd(1, sim_par.prob); 
           elseif stim_pres(t) == 4 %No-go-win (NGW)
              reinforcement(t,1) = sim_par.reward*binornd(1, 1-sim_par.prob);
           end
        end
        
    end 
           
    
end 

% Re-create Plots from the Paper (Figure 2.A-D) 
% plot(ActionProb) 

% Plot Behavior depending on parameter settings 
% .... 







