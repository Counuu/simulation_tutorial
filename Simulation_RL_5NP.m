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
    Q = zeros(sim_par.n_cond,sim_par.n_actions);
    ActionWeight_go = zeros(sim_par.n_trial_cond,sim_par.n_cond,sim_par.n_actions);
    ActionWeight_nogo = zeros(sim_par.n_trial_cond,sim_par.n_cond,sim_par.n_actions);
    ActionProb = ones(sim_par.n_trial_cond,sim_par.n_cond,sim_par.n_actions)/2;
    
    % ==== Stimulus conditions ============================================
    % Go through all stimulus conditions
    for s = 1:sim_par.n_cond
        softmaxval = 0;
        % ==== Learning for each trial ====================================
        % Go through each trial of current stimulus conditon
        for t = 1:sim_par.n_trial_cond
        
            % First trial with initial settings (no previous knowledge)
            if t == 1 
               ActionChoice(t,s) = binornd(1, ActionProb(t,s)); % 0 = No-Go, 1 = Go 
               
            % All subsequent trials
            else
                % For each possible action 
                for a = 1:sim_par.n_actions
                    
                    % Q-update: Rescorla-Wagner update       
                    % > separate parameter for sensitivity for reward (gamma) and punishment (delta) 
                    % > reinforcement t-1, as they refer to the previous trial 
                    if reinforcement(t-1,s) == 1 %rewarded
                       Q(s,a) = Q(s,a) +  sim_par.alpha* ((sim_par.gamma * reinforcement(t-1,s)) - Q(s,a)); 
                    elseif reinforcement(t-1,s) == -1 %punished
                       Q(s,a) = Q(s,a) +  sim_par.alpha* ((sim_par.delta * reinforcement(t-1,s)) - Q(s,a));  
                    elseif reinforcement(t-1,s) == 0 %nothing
                       Q(s,a) = Q(s,a) +  sim_par.alpha* ( - Q(s,a));
                    end 
                    
                
                    % Action Weight for go and no-go 
                    ActionWeight_go(t,s,a) = Q(s,a) + sim_par.zeta ;
                    ActionWeight_nogo(t,s,a) =  Q(s,a) ;

                    % Action Probability for Go (softmax function)
                    n(t,1:2) = [ActionWeight_go(t,s,a); ActionWeight_nogo(t,s,a)]; 
                    softmaxval = softmax(n); 


                    % Action Probability for Go Action 
                    ActionProb(t,s,a) = softmaxval(1,a) * (1 - sim_par.xi) + (sim_par.xi/2);
                
                    % Action Choice Go
                    ActionChoice(t,s) = binornd(1, ActionProb(t,s,a));   

                end 
            end 

            % Reinforcement Value Calculation: depending on Action & Stimulus (s)
            % Reinforcement values: +1 for reward, 0 for nothing and -1 for punishment 
        
            if ActionChoice(t,s) == 0 % No-go action 
               if s == 1 %Go-to-avoid (GA)
                  reinforcement(t,s) = sim_par.punish *binornd(1, sim_par.prob); 
               elseif s == 2 %Go-to-win (GW) 
                  reinforcement(t,s) = sim_par.reward*binornd(1, 1-sim_par.prob); 
               elseif s == 3 %No-go-to-avoid (NGA)
                  reinforcement(t,s) = sim_par.punish *binornd(1, 1-sim_par.prob); 
               elseif s == 4 %No-go-win (NGW)
                  reinforcement(t,s) = sim_par.reward*binornd(1, sim_par.prob);
               end 

            elseif ActionChoice(t,s) == 1 %1: Go action
               if s == 1 %Go-to-avoid (GA)
                  reinforcement(t,s) = sim_par.punish *binornd(1, 1-sim_par.prob); 
               elseif s == 2 %Go-to-win (GW) 
                  reinforcement(t,s) = sim_par.reward*binornd(1, sim_par.prob); 
               elseif s == 3 %No-go-to-avoid (NGA)
                  reinforcement(t,s) = sim_par.punish *binornd(1, sim_par.prob); 
               elseif s == 4 %No-go-win (NGW)
                  reinforcement(t,s) = sim_par.reward*binornd(1, 1-sim_par.prob);
               end
            end
            
        end 
    end 
           
    
end 

% Re-create Plots from the Paper (Figure 2.A-D) 
plot(ActionProb(:,:,1))

% Plot Behavior depending on parameter settings 
% .... 






