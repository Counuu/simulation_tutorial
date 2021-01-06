%% Simulation of a Reinforcement Learning Model (5NP)
% This code simulates a 5-parameter RL Model, as described in 
% Guitart-Masip et al. (2012)
% Computational Psychiatry Seminar WS 20/21 Dr. Nils Kroemer
% Code written: Kirsti, Sophie, Corinna, Xin 
% Based on RL_tut.m script written by Dr. Nils Kroemer

%% Prepare workspace
clear all
close all

%% Initialise parameters 
% sim_par is a structure containing all settings for the simulation

%Experiment settings
sim_par.p_win = 0.8; %win probability
sim_par.p_lose = 0.2; %lose probability
sim_par.n_trials = 240; %number of trials (total)
sim_par.n_part = 50;   %number of simulated participants
sim_par.n_cond = 4; %number of conditions 
%sim_par.prob = 0.8; %probability to reinforce (vs. nothing)
sim_par.reward = 1; %valence of reward
sim_par.punish = -1; %valence of punishment 
sim_par.nothing = 0; %valence of no reward/no punish
sim_par.n_trial_cond = sim_par.n_trials/sim_par.n_cond; %nr of trials per condition
sim_par.n_actions = 2; %nr of possible actions (go, no-go)

%RL settings
sim_par.alpha = 0.1;    %learning rate of the simulated agent
% sim_par.beta  = 2;      %inverse temperature of the simulated agent
sim_par.xi  = 0.05;      %lapse of the simulated agent
sim_par.gamma  = 4;     %reward sensitivity of the simulated agent
sim_par.delta  = 6;     %punishment sensitivty of the simulated agent
%sim_par.epsilon  = 0;   %approach-avoidance bias of the simulated agent
sim_par.zeta  = 0.2;      %action bias of the simulated agent

%There are 4 conditions: GA (go to avoid), GW (go to win), NGA (no go to
%avoid) and NGW (no go to win)
textConds = {'GA','GW','NGA','NGW'};

%Create results folder if not existent to store plots and data
mkdir([pwd '\results'], '\plots')
mkdir([pwd '\results'], '\data')

%% Simulation loop for Reinforcement Learning
for subj = 1:sim_par.n_part %Runs for specified number of simulated participants
    
    lf_results = cell(sim_par.n_trials,7); %longformat array to store the results
    
    % ==== Stimulus conditions ============================================
    % for each experimental run stimulus conditions are
    % pseudorandomly determind (equally often, but randomly shuffled)
    each_cond = sim_par.n_trials/sim_par.n_cond; 
    stim_pres = [ones(each_cond,1);2*ones(each_cond,1);3*ones(each_cond,1);4*ones(each_cond,1)]; 
    stim_pres = stim_pres(randperm(length(stim_pres))); 
    
    %Preallocation of variables to use them for storage later
    Q = zeros(sim_par.n_cond,sim_par.n_actions); % Q-Values
    ActionChoice = NaN(sim_par.n_trials,1); % which action was chosen (go vs no-go)
    ActionChoiceProb = NaN(sim_par.n_trials,2); % action choice probability for each action option (go & no-go)
    goodChoice = NaN(sim_par.n_trials,1); % which action would be the correct choice
    reward = NaN(sim_par.n_trials,1); % reward outcome (punishment, reward, or nothing)
    RPE = zeros(sim_par.n_trials,2); % predicition errors
    
    % Go through all trials
    for trial = 1:sim_par.n_trials
        
        cond = stim_pres(trial); % currently presented stimulus (= condition) 
        
        % ==== Learning for each trial ====================================

        Go = exp(Q(cond,2)+sim_par.zeta); % Action weight for go
        NoGo = exp(Q(cond,1)); % Action weight for no-go
        ActionChoiceProb(trial,1) = (Go / (Go + NoGo))*(1-sim_par.xi)+(sim_par.xi/2); %Go probabiliy (softmax function) 
        ActionChoiceProb(trial,2) = (NoGo / (Go + NoGo))*(1-sim_par.xi)+(sim_par.xi/2); %No-go probability (softmax function)

        % Action Choice 
        if rand < ActionChoiceProb(trial,1)
           ActionChoice(trial,1) = 2; %picks go
        else
           ActionChoice(trial,1) = 1; %picks no-go
        end              


        % Define good Choice (depends on condition)
        if cond == 1 %GA condition
            goodChoice(trial,1) = 2; %Go
        elseif cond == 2 %GW condition
            goodChoice(trial,1) = 2; %Go
        elseif cond == 3 %NGA condition
            goodChoice(trial,1) = 1; %No-Go
        elseif cond == 4 %NGW condition
            goodChoice(trial,1) = 1; %No-Go
        end
        
        % calculate reward depending on choice(correct vs wrong) and
        % condition (reward vs punishment)
        if ActionChoice(trial,1) == goodChoice(trial,1) % good option chosen
            if cond == 2 || cond == 4 % winning condition
                reward(trial,1) = double(rand < sim_par.p_win); % 80% chance of +1 reward
            elseif cond == 1 || cond == 3 % avoiding condition
                reward(trial,1) = -(double(rand < 1-sim_par.p_win)); % 20% chance of -1 reward (=punishment)
            end
        else % bad action chosen
            if cond == 2 || cond == 4 %winning condition
                reward(trial,1) = double(rand < sim_par.p_lose); % 20% chance of +1 reward
            elseif cond == 1 || cond == 3 %avoiding condition
                reward(trial,1) = -(double(rand < 1-sim_par.p_lose)); % 80% chance of -1 reward (=punishment)
            end
        end

        % calculate RPE based on the reward
        if reward(trial,1) == 1 %reward
            RPE(trial,ActionChoice(trial,1)) = sim_par.gamma*reward(trial,1) - Q(cond,ActionChoice(trial,1)); % gamma is reward sensitivity
        elseif reward(trial,1) == -1 %punishment
            RPE(trial,ActionChoice(trial,1)) = sim_par.delta*reward(trial,1) - Q(cond,ActionChoice(trial,1)); % delta is punishment sensitivity 
        elseif reward(trial,1) == 0 %nothing
            RPE(trial,ActionChoice(trial,1)) = - Q(cond,ActionChoice(trial,1));
        end

        % update Q-values for both possible actions, alpha is the learning rate 
        Q(cond,1) = Q(cond,1) + sim_par.alpha * RPE(trial,1); %no-go
        Q(cond,2) = Q(cond,2) + sim_par.alpha * RPE(trial,2); %go 

       %store longformat results
       condition_text = textConds{cond};
       lf_results(trial,:) = [trial, {condition_text}, ActionChoice(trial,1), goodChoice(trial,1), ActionChoiceProb(trial,1), ActionChoiceProb(trial,2), Q(cond,1)];

    end   
    
    %summarize all results for this simulated participant in a table
    conditions = categorical(lf_results(:,2),{'GA','GW','NGA', 'NGW'});
    results_longformat_table = table(lf_results(:,1),conditions,lf_results(:,3),lf_results(:,4),lf_results(:,5),lf_results(:,6),lf_results(:,7),'VariableNames',{'trial','condition','ActionChoice','goodChoice','goProb','nogoProb','Q'});

    %plot go probabilities for each condition
    figureResults = figure('Name','Go-probability development for all conditions');
        for k = 1:4
            iCond = textConds{k};
            plot(1:60,cell2mat(results_longformat_table.goProb(results_longformat_table.condition==iCond)),'DisplayName',iCond);
            hold on
        end
        legend
        title('Go-probability development for each condition')
        xlabel('trials')
        ylabel('Go probability')    
        
    % save plot and simulation data    
    saveas(figureResults,[pwd '\results\plots\goprob_subj_' num2str(subj) '.png'])    
    save([pwd '\results\data\simulation_subj_' num2str(subj)], 'results_longformat_table') 
    
    close all
end 

%% Summarize go-probabilities for all participants depending on condition
%Preallocation of variables
allSubj_GA = NaN(sim_par.n_trial_cond,sim_par.n_part); %GA condition
allSubj_GW = NaN(sim_par.n_trial_cond,sim_par.n_part); % GW condition
allSubj_NGA = NaN(sim_par.n_trial_cond,sim_par.n_part); % NGA condition
allSubj_NGW = NaN(sim_par.n_trial_cond,sim_par.n_part); % NGW condition

%read all results from the data folder for each simulated participant
for subj = 1:sim_par.n_part
    load([pwd '\results\data\simulation_subj_' num2str(subj) '.mat'], 'results_longformat_table')
    allSubj_GA(:,subj)= cell2mat(results_longformat_table.goProb(results_longformat_table.condition=='GA'));
    allSubj_GW(:,subj)= cell2mat(results_longformat_table.goProb(results_longformat_table.condition=='GW'));
    allSubj_NGA(:,subj)= cell2mat(results_longformat_table.goProb(results_longformat_table.condition=='NGA'));
    allSubj_NGW(:,subj)= cell2mat(results_longformat_table.goProb(results_longformat_table.condition=='NGW'));
end

%calculate mean of go-probability over all participants for each trial
allSubj_GA(:,sim_par.n_part+1) = mean(allSubj_GA(:,1:sim_par.n_part),2,'omitnan');
allSubj_GW(:,sim_par.n_part+1) = mean(allSubj_GW(:,1:sim_par.n_part),2,'omitnan');
allSubj_NGA(:,sim_par.n_part+1) = mean(allSubj_NGA(:,1:sim_par.n_part),2,'omitnan');
allSubj_NGW(:,sim_par.n_part+1) = mean(allSubj_NGW(:,1:sim_par.n_part),2,'omitnan');

%plot means of go probabilities for each condition
figureResultsAllSubj = figure('Name','Mean results for all conditions over all participants');
    plot(1:sim_par.n_trial_cond, allSubj_GA(:,sim_par.n_part+1),'DisplayName','GA')
    hold on
    plot(1:sim_par.n_trial_cond, allSubj_GW(:,sim_par.n_part+1),'DisplayName','GW')
    plot(1:sim_par.n_trial_cond, allSubj_NGA(:,sim_par.n_part+1),'DisplayName','NGA')
    plot(1:sim_par.n_trial_cond, allSubj_NGW(:,sim_par.n_part+1),'DisplayName','NGW')
    legend
    title('Mean results for each condition')
    xlabel('trials')
    ylabel('Go probability')

% save plot
saveas(figureResultsAllSubj,[pwd '\results\plots\go_prob_allsubj.png'])     


