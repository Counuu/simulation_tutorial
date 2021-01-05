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
textConds = {'GA','GW','NGA','NGW'};

for i = 1:sim_par.n_part %For now one sim, later with different parameter settings
    
    data = struct;
    lf_results = [];
    % ==== Stimulus conditions ============================================
    % for each experimental run stimulus conditions are
    % pseudorandomly determind (equally often, but randomly shuffled)
    each_cond = sim_par.n_trials/sim_par.n_cond; 
    stim_pres = [ones(each_cond,1);2*ones(each_cond,1);3*ones(each_cond,1);4*ones(each_cond,1)]; 
    data.stim_pres = stim_pres(randperm(length(stim_pres))); 
    
    % ==== Initialise values when no experience exists ====================
    data.Q = zeros(sim_par.n_cond,sim_par.n_actions);
%     data.ActionWeight_go = {};%NaN(sim_par.n_trial_cond,sim_par.n_cond,sim_par.n_actions);
%     data.ActionWeight_nogo = {};%NaN(sim_par.n_trial_cond,sim_par.n_cond,sim_par.n_actions);
    data.ActionProb = {ones(2,1)/2;ones(2,1)/2;ones(2,1)/2;ones(2,1)/2};%ones(sim_par.n_trial_cond,sim_par.n_cond,sim_par.n_actions)/2;
    data.ActionChoice = NaN(sim_par.n_trials,1);
    
%     softmaxval = 0;
    StimCounter = ones(4,1); 
    
    % Go through all trials
    for trial = 1:sim_par.n_trials
        
        cond = data.stim_pres(trial); % currently presented stimulus (condition) 
        
        % ==== Learning for each trial ====================================
        condTrialNr = StimCounter(cond,1);
            % First trial with initial settings (no previous knowledge)
            if condTrialNr  == 1 
               data.ActionChoice(trial,1) = binornd(1, data.ActionProb{cond}(condTrialNr,1))+1; % 1 = No-Go, 2 = Go 
               
            % All subsequent trials
            else
                data.ActionWeight{cond}(condTrialNr,1:2) = [data.Q(cond,1) + sim_par.zeta , data.Q(cond,2)]; % Action weight for go and no-go 
                %data.ActionProb{cond}(condTrialNr,1) = (exp(data.ActionWeight{cond}(condTrialNr,1))/sum(exp(data.ActionWeight{cond}(condTrialNr,1:2)),'all'))* (1 - sim_par.xi) + (sim_par.xi/2); % Calculate Go probability (softmax and irreducible noise (xi))
                data.ActionProb{cond}(condTrialNr,1) = (exp(data.ActionWeight{cond}(condTrialNr,1))/sum(exp(data.Q(cond,:)),'all'))* (1 - sim_par.xi) + (sim_par.xi/2); % Calculate Go probability (softmax and irreducible noise (xi))   
                %data.ActionProb{cond}(condTrialNr,2) = (exp(data.ActionWeight{cond}(condTrialNr,2))/sum(exp(data.ActionWeight{cond}(condTrialNr,1:2)),'all'))* (1 - sim_par.xi) + (sim_par.xi/2); % Calculate Go probability (softmax and irreducible noise (xi))
                maxProb = double(data.ActionProb{cond}(condTrialNr,:)==max(data.ActionProb{cond}(condTrialNr,:)));
                %data.ActionChoice(trial,1)= maxProb(1)+1;
                data.ActionChoice(trial,1) = binornd(1,data.ActionProb{cond}(condTrialNr,1))+1; % Action Choice: 1 = No-go, 2= Go
            end 
            
            StimCounter(cond,1) = StimCounter(cond,1)+1;

            % Reinforcement Value Calculation: depending on Action & Stimulus (s)
            % Reinforcement values: +1 for reward, 0 for nothing and -1 for punishment 
        
            if data.ActionChoice(trial,1) == 1 % No-go action 
               if cond == 1 %Go-to-avoid (GA)
                  data.reinforcement(trial,1) = sim_par.punish *binornd(1, sim_par.prob); 
               elseif cond == 2 %Go-to-win (GW) 
                  data.reinforcement(trial,1) = sim_par.reward*binornd(1, 1-sim_par.prob); 
               elseif cond == 3 %No-go-to-avoid (NGA)
                  data.reinforcement(trial,1) = sim_par.punish *binornd(1, 1-sim_par.prob); 
               elseif cond == 4 %No-go-win (NGW)
                  data.reinforcement(trial,1) = sim_par.reward*binornd(1, sim_par.prob);
               end 
            elseif data.ActionChoice(trial,1) == 2 %1: Go action
               if cond == 1 %Go-to-avoid (GA)
                  data.reinforcement(trial,1) = sim_par.punish *binornd(1, 1-sim_par.prob); 
               elseif cond == 2 %Go-to-win (GW) 
                  data.reinforcement(trial,1) = sim_par.reward*binornd(1, sim_par.prob); 
               elseif cond == 3 %No-go-to-avoid (NGA)
                  data.reinforcement(trial,1) = sim_par.punish *binornd(1, sim_par.prob); 
               elseif cond == 4 %No-go-win (NGW)
                  data.reinforcement(trial,1) = sim_par.reward*binornd(1, 1-sim_par.prob);
               end                          
            end
            
            % Q-update: Rescorla-Wagner update
           % > separate parameter for sensitivity for reward (gamma) and punishment (delta) 
           % > reinforcement t-1, as they refer to the previous trial 
           if data.reinforcement(trial,1) == sim_par.reward  %rewarded
              data.Q(cond,data.ActionChoice(trial,1)) = data.Q(cond,data.ActionChoice(trial,1)) +  sim_par.alpha* ((sim_par.gamma * data.reinforcement(trial,1)) - data.Q(cond,data.ActionChoice(trial,1))); 
           elseif data.reinforcement(trial,1) == sim_par.punish  %punished
              data.Q(cond,data.ActionChoice(trial,1)) = data.Q(cond,data.ActionChoice(trial,1)) +  sim_par.alpha* ((sim_par.delta * data.reinforcement(trial,1)) - data.Q(cond,data.ActionChoice(trial,1)));  
           elseif data.reinforcement(trial,1) == sim_par.nothing %nothing
              data.Q(cond,data.ActionChoice(trial,1)) = data.Q(cond,data.ActionChoice(trial,1)) +  sim_par.alpha* ( - data.Q(cond,data.ActionChoice(trial,1)));
           end 
           
           condition_text = textConds{cond};
           lf_results = [lf_results; trial, {condition_text}, data.ActionChoice(trial,1), data.reinforcement(trial,1),max(data.ActionProb{cond}(condTrialNr,:))];
            
    end       
                                
end 

conditions = categorical(lf_results(:,2),{'GA','GW','NGA', 'NGW'});
results_longformat_table = table(lf_results(:,1),conditions,lf_results(:,3),lf_results(:,4),lf_results(:,5),'VariableNames',{'trial','condition','action','reinforcement','go_prob'});

figureResults = figure('Name','Results for all conditions');
    for i = 1:4
        iCond = textConds{i};
        plot([1:60],cell2mat(results_longformat_table.go_prob(results_longformat_table.condition==iCond)),'DisplayName',iCond);
        hold on
    end
    legend
    title('Results for each condition')
    xlabel('trials')
    ylabel('Go probability')







