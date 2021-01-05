%% Simulation of a Reinforcement Learning Model (5NP)
% This code simulates a 5-parameter RL Model, as described in 
% Guitart-Masip et al. (2012)
% Computational Psychiatry Seminar WS 20/21 Dr. Nils Kroemer
% Code written: Kirsti, Sophie, Corinna, Xin 

clear all

%% Initialise paramters 
% sim_par is a structure containing all settings for the simulation

%Experiment settings
sim_par.n_trials = 240; %number of trials (total)
sim_par.n_part = 1;   %number of simulated participants
sim_par.n_cond = 4; %number of conditions 
sim_par.prob = 0.8; %probability to reinforce (vs. nothing)
sim_par.reward = 1; %valence of reward
sim_par.punish = -1; %valence of punishment 
sim_par.nothing = 0; %valence of no reward/no punish
sim_par.n_trial_cond = sim_par.n_trials/sim_par.n_cond; %nr of trials per condition
sim_par.n_actions = 2; %nr of possible actions (go, no-go)

%% Simulation of Agents Behavior using Reinforcement Learning 

% RL parameters: later inspect different model paramter outcomes! 
sim_par.alpha = 0.1;    %learning rate of the simulated agent
sim_par.xi  = 0.5;      %lapse of the simulated agent
sim_par.gamma  = 1;     %reward sensitivity of the simulated agent
sim_par.delta  = 1;     %punishment sensitivty of the simulated agent
%sim_par.epsilon  = 0;   %approach-avoidance bias of the simulated agent
sim_par.zeta  = 0.2;      %action bias of the simulated agent

% Preallocation of variables to increase loop speed 
 %Q
% ActionWeight_go
% ActionWeight_nogo
% ActionProb
% ActionChoice
% reinforcement 


for i = 1:sim_par.n_part %For now one sim, later with different parameter settings
    
    results = struct;
    
    % ==== Stimulus conditions ============================================
    % for each experimental run stimulus conditions are
    % pseudorandomly determind (equally often, but randomly shuffled)
    each_cond = sim_par.n_trials/sim_par.n_cond; 
    stim_pres = [ones(each_cond,1);2*ones(each_cond,1);3*ones(each_cond,1);4*ones(each_cond,1)]; 
    results.stim_pres = stim_pres(randperm(length(stim_pres))); 
    
    % ==== Initialise values when no experience exists ====================
    results.Q = zeros(sim_par.n_cond,sim_par.n_actions);
    results.ActionWeight_go = {};%NaN(sim_par.n_trial_cond,sim_par.n_cond,sim_par.n_actions);
    results.ActionWeight_nogo = {};%NaN(sim_par.n_trial_cond,sim_par.n_cond,sim_par.n_actions);
    results.ActionProb = {};%ones(sim_par.n_trial_cond,sim_par.n_cond,sim_par.n_actions)/2;
    results.ActionChoice = zeros(sim_par.n_trials,1);
    
%     softmaxval = 0;
    StimCounter = ones(4,1); 
    
    % Go through all trials
    for trial = 1:sim_par.n_trials
        
        cond = results.stim_pres(trial); % currently presented stimulus (condition) 
        
        % ==== Learning for each trial ====================================
        
            % First trial with initial settings (no previous knowledge)
            if StimCounter(cond,1) == 1 
               results.ActionChoice(trial,1) = binornd(1, results.ActionProb{cond,1}(trial,1)); % 0 = No-Go, 1 = Go 
               
            % All subsequent trials
            else
                for ActionOption = 1:2 %Go and No-Go
                    % Action Weight for go and no-go 
                    results.ActionWeight{cond,ActionOption}(trial,1) = results.Q(cond,ActionOption) + sim_par.zeta ; %Action Weight Go
                    results.ActionWeight{cond,ActionOption}(trial,2) =  results.Q(cond,ActionOption) ;%Action Weight No-Go

                    % Action Probability for Go and No-Go(softmax function)
                    n(trial,1:2) = [results.ActionWeight{cond,ActionOption}(:,1); results.ActionWeight{cond,ActionOption}(:,2)]; 
                    softmaxval = softmax(n); 

                    % Action Probability for Go Action 
                    ActionProb(cond,ActionOption) = softmaxval(1,ActionOption) * (1 - sim_par.xi) + (sim_par.xi/2);
                 
                    % Action Choice Go
                    ActionChoice(trial,1) = binornd(1, ActionProb(cond,ActionOption));   
                end 
                 
            end 
            
            StimCounter(cond,1) = StimCounter(cond,1)+1;

            % Reinforcement Value Calculation: depending on Action & Stimulus (s)
            % Reinforcement values: +1 for reward, 0 for nothing and -1 for punishment 
        
            if results.ActionChoice(trial,1) == 0 % No-go action 
               if cond == 1 %Go-to-avoid (GA)
                  results.reinforcement(trial,1) = sim_par.punish *binornd(1, sim_par.prob); 
               elseif cond == 2 %Go-to-win (GW) 
                  results.reinforcement(trial,1) = sim_par.reward*binornd(1, 1-sim_par.prob); 
               elseif cond == 3 %No-go-to-avoid (NGA)
                  results.reinforcement(trial,1) = sim_par.punish *binornd(1, 1-sim_par.prob); 
               elseif cond == 4 %No-go-win (NGW)
                  results.reinforcement(trial,1) = sim_par.reward*binornd(1, sim_par.prob);
               end 
               
                

%                % Q-update: Rescorla-Wagner update       
%                % > separate parameter for sensitivity for reward (gamma) and punishment (delta) 
%                % > reinforcement t-1, as they refer to the previous trial 
%                if reinforcement(trial) == sim_par.reward  %rewarded
%                   Q(cond,1) = Q(cond,1) +  sim_par.alpha* ((sim_par.gamma * reinforcement(trial)) - Q(cond,1)); 
%                elseif reinforcement(trial) == sim_par.punish  %punished
%                   Q(cond,1) = Q(cond,1) +  sim_par.alpha* ((sim_par.delta * reinforcement(trial)) - Q(cond,1));  
%                elseif reinforcement(trial) == sim_par.nothing %nothing
%                   Q(cond,1) = Q(cond,1) +  sim_par.alpha* ( - Q(cond,1));
%                end 

            elseif results.ActionChoice(trial,1) == 1 %1: Go action
               if cond == 1 %Go-to-avoid (GA)
                  results.reinforcement(trial,1) = sim_par.punish *binornd(1, 1-sim_par.prob); 
               elseif cond == 2 %Go-to-win (GW) 
                  results.reinforcement(trial,1) = sim_par.reward*binornd(1, sim_par.prob); 
               elseif cond == 3 %No-go-to-avoid (NGA)
                  results.reinforcement(trial,1) = sim_par.punish *binornd(1, sim_par.prob); 
               elseif cond == 4 %No-go-win (NGW)
                  results.reinforcement(trial,1) = sim_par.reward*binornd(1, 1-sim_par.prob);
               end
               
%                % Q-update: Rescorla-Wagner update       
%                % > separate parameter for sensitivity for reward (gamma) and punishment (delta) 
%                % > reinforcement t-1, as they refer to the previous trial 
%                if reinforcement(trial) == sim_par.reward  %rewarded
%                   Q(cond,2) = Q(cond,2) +  sim_par.alpha* ((sim_par.gamma * reinforcement(trial)) - Q(cond,2)); 
%                elseif reinforcement(trial) == sim_par.punish  %punished
%                   Q(cond,2) = Q(cond,2) +  sim_par.alpha* ((sim_par.delta * reinforcement(trial)) - Q(cond,2));  
%                elseif reinforcement(trial) == sim_par.nothing %nothing
%                   Q(cond,2) = Q(cond,2) +  sim_par.alpha* ( - Q(cond,2));
%                end 
               
            end
            
            % Q-update: Rescorla-Wagner update
           % > separate parameter for sensitivity for reward (gamma) and punishment (delta) 
           % > reinforcement t-1, as they refer to the previous trial 
           if results.reinforcement(trial,1) == sim_par.reward  %rewarded
              Q(cond,results.ActionChoice(trial,1)) = Q(cond,results.ActionChoice(trial,1)) +  sim_par.alpha* ((sim_par.gamma * results.reinforcement(trial,1)) - Q(cond,results.ActionChoice(trial,1))); 
           elseif results.reinforcement(trial) == sim_par.punish  %punished
              Q(cond,results.ActionChoice(trial,1)) = Q(cond,results.ActionChoice(trial,1)) +  sim_par.alpha* ((sim_par.delta * results.reinforcement(trial,1)) - Q(cond,results.ActionChoice(trial,1)));  
           elseif results.reinforcement(trial) == sim_par.nothing %nothing
              Q(cond,results.ActionChoice(trial,1)) = Q(cond,results.ActionChoice(trial,1)) +  sim_par.alpha* ( - Q(cond,results.ActionChoice(trial,1)));
           end 
           
           
            lf_results = [lf_results; trial, cond, results.ActionChoice(trial,1), results.reinforcement(trial,1)
            
    end       

            
         
     
           
    
end 

% Re-create Plots from the Paper (Figure 2.A-D) 
plot(ActionProb(:,:,1))

% Plot Behavior depending on parameter settings 
% .... 






