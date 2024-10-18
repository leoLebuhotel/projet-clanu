    clear all;
close all;
clc;
addpath('../');

%%%%%%%%%%%%%%%%%%%%% data base %%%%%%%%%%%%%%%%%%%%
m = 20;
nc = 4;
x1 = [1 + 0.15*randn(1,m),-1/2 + 0.15*randn(1,m), 0.15*randn(1,m),2+0.15*randn(1,m)];
x2 = [1 + 0.15*randn(1,m), 0.15*randn(1,m), -1/2 + 0.15*randn(1,m),-2 + 0.15*randn(1,m)];
y1 = [ones(1,m),zeros(1,m),zeros(1,m),zeros(1,m)];
y2 = [zeros(1,m),ones(1,m),zeros(1,m),zeros(1,m)];
y3 = [zeros(1,m),zeros(1,m),ones(1,m),zeros(1,m)];
y4 = [zeros(1,m),zeros(1,m),zeros(1,m),ones(1,m)];

database.X_train = [x1;x2];
database.Y_train = [y1;y2;y3;y4];




%-- Display training database
visu.display_point_cloud(database,'Training dataset');


%-- Build a model with a n_h-dimensional hidden layer
num_iterations = 30000;
learning_rate = 0.15;
print_cost = true;
nX = size(database.X_train,1);
layers_dims = [nX,5,4,3,nc];
[parameters,costs] = L_layers_nn.model(database, layers_dims, num_iterations, learning_rate, print_cost);


%-- Display decision boundary
visu.display_decision_boundary(database,parameters);


%-- Compute accuracy
X_train = database.X_train;
Y_train = database.Y_train;
Y_prediction_train = L_layers_nn.predict(parameters, X_train);



%% commentaires
%désormais, pour le calcul du coût on doit prendre en compte chaque ligne
%du vecteur Y (et donc chaque classe) contrairement à avant où on n'avait
%que deux classes possibles. On utilise donc une somme, et on somme le coût
%pour chaque ligne (donc chaque classe), plutôt que de faire un calcul
%matriciel plus simple

%la fonction predict a été modifiée afin d'assurer des prédictions qui vont
%entre 0 et nc, nc étant le nombre de classes, plutôt qu'entre 0 et 1. Dans
%la première partie de la fonction on peut observer cet algorithme de
%normalisation permettant d'obtenir cela. Dans l'exemple avec m = 3 et 3
%classes on a Ypredicition = 1 1 1 2 2 2 3 3 3 ! Et on a donc nos 3 classes
%distinctes

