function [database] = structure_database(pd_a,pd_c,pd_s,t1_a,t1_c,t1_s,t2_a,t2_c,t2_s,nclasse)

img = [pd_a pd_c pd_s t1_a t1_c t1_s t2_a t2_c t2_s]; % 900 images

random = randperm(100);


X_tr = [];
X_v = [];
X_t = [];
Y_tr = [];
Y_v = [];
Y_t = [];


for i = 0:nclasse-1
for k = 1:80
    X_tr = [X_tr img{100*(i)+random(k)}(:)];

end

for k = 81:90
    X_v = [X_v img{100*(i)+random(k)}(:)];
end

for k = 91:100
    X_t = [X_t img{100*(i)+random(k)}(:)];
end

Y_tr = [Y_tr; zeros(1,i*80) ones(1,80) zeros(1,(nclasse-i-1)*80)];
Y_v = [Y_v ; zeros(1,i*10) ones(1,10) zeros(1,(nclasse-i-1)*10)];
Y_t = [Y_t; zeros(1,i*10) ones(1,10) zeros(1,(nclasse-i-1)*10)];
end

database.X_train = X_tr;
database.X_valid = X_v;
database.X_test = X_t;

database.Y_train = Y_tr;
database.Y_valid = Y_v;
database.Y_test = Y_t;

database.num_px = 64;

