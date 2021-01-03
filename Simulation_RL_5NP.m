%% Simulation of a Reinforcement Learning Model (5NP)
% This code simulates a 5-parameter RL Model, as described in 
% Guitart-Masip et al. (2012)
% Computational Psychiatry Seminar WS 20/21 Dr. Nils Kroemer
% Code written: Kirsti, Sophie, Corinna, Xin
% hello from Sophie :)
%% Initialise paramters 
% sim_par is a structure containing all settings for the simulation

%Experiment settings
sim_par.n_trials = 60; %number of trials
sim_par.n_part = 1;   %number of simulated participants
sim_par.n_cond = 4; %number of conditions 
sim_par.prob = 0.8; %probability to reinforce (vs. nothing)
sim_par.reward = 1; %valence of reward
sim_par.punish = -1; %valence of punishment 

%% Simulation of Agents Behavior using Reinforcement Learning 

%RL parameters: inspect different model paramter outcomes! 
sim_par.alpha = 0.1;    %learning rate of the simulated agent
sim_par.xi  = 0.5;      %lapse of the simulated agent
sim_par.gamma  = 1;     %reward sensitivity of the simulated agent
sim_par.delta  = 1;     %punishment sensitivty of the simulated agent
%sim_par.epsilon  = 0;   %approach-avoidance bias of the simulated agent
sim_par.zeta  = 1;      %action bias of the simulated agent

% Preallocation of variables to increase loop speed 

for i = 1:sim_par.n_part
    
    %Stimulus conditions
    %for each experimental run stimulus conditions are
    %pseudorandomly determind (equally often, but randomly shuffled)
    each_cond = sim_par.n_trials/sim_par.n_cond; 
    stim_pres = [ones(each_cond,1);2*ones(each_cond,1);3*ones(each_cond,1);4*ones(each_cond,1)]; 
    stim_pres = stim_pres(randperm(length(stim_pres))); 
    
    for t = 1:sim_par.n_trials
        
        %Learning Model
        
        %First trial using initial settings 
        if t == 1 
           Q(t) = 0; %initial Q-value is set to zero 
           
           % On the first trial the action (go vs. nogo) is random
           ActionProb(t,1) = 0.5;
           
           % binord generates random numbers from binomial distribution, nr trials n, prob of success for each trial p.
           ActionChoice(t,1) = binornd(1, ActionProb(t,1));       
        
        %All subsequent trials
        else
           
            %Q-update: Rescorla-Wagner update       
            % separate parameter for sensitivity for reward and punishment 
            if reinforcement == 1 
               Q(t) = Q(t-1) +  sim_par.alpha* ((sim_par.gamma * reinforcement) - Q(t-1)); 
            elseif reinforcement == -1 
               Q(t) = Q(t-1) +  sim_par.alpha* ((sim_par.delta * reinforcement) - Q(t-1));  
            end 
            
            %Action Weight for go and no-go 
            ActionWeight.go(t,1) = Q(t) + sim_par.zeta ;
            ActionWeight.nogo(t,1) =  Q(t) ;

            %Action Probability (softmax function)
            n = [ActionWeight.go; ActionWeight.nogo];
            a = exp(n)/sum(exp(n)); %this is softmax(n) 
            subplot(2,1,1), bar(n), ylabel('n')
            subplot(2,1,2), bar(a), ylabel('a')
            
            % Calculate Action Probability for Go Action 
            ActionProb(t,1) = a(1) * (1 - sim_par.xi) + (sim_par.xi/2);
            
            % Make Action Choice 
            ActionChoice(t,1) = binornd(1, ActionProb(t,1));   
       
        end 
        

        %Agent choses action, now reinforcement is determined, given stimulus
        if ActionChoice == 0 % No-go action 
           if stim_pres(1) == 1 %Go-to-avoid (GA)
              reinforcement = sim_par.punish *binornd(1, sim_par.prob); 
           elseif stim_pres(1) == 2 %Go-to-win (GW) 
              reinforcement = sim_par.reward*binornd(1, 1-sim_par.prob); 
           elseif stim_pres(1) == 3 %No-go-to-avoid (NGA)
              reinforcement = sim_par.punish *binornd(1, 1-sim_par.prob); 
           elseif stim_pres(1) == 4 %No-go-win (NGW)
              reinforcement = sim_par.reward*binornd(1, sim_par.prob);
           end 
   
  
        elseif ActionChoice == 1 %1: Go action
           if stim_pres(1) == 1 %Go-to-avoid (GA)
              reinforcement = sim_par.punish *binornd(1, 1-sim_par.prob); 
           elseif stim_pres(1) == 2 %Go-to-win (GW) 
              reinforcement = sim_par.reward*binornd(1, sim_par.prob); 
           elseif stim_pres(1) == 3 %No-go-to-avoid (NGA)
              reinforcement = sim_par.punish *binornd(1, sim_par.prob); 
           elseif stim_pres(1) == 4 %No-go-win (NGW)
              reinforcement = sim_par.reward*binornd(1, 1-sim_par.prob);
           end
        end
        
    end 
           
    
end 






