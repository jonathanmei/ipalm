%test_ipalm
%solve sparse pca problem
% min 1/2*|| Y-UV ||_F^2 + lam*||U||_1 + lam/2*||V||_F^2

M=33; %dimensions
N=25; %dimensions
r=6; %rank of matrix


%subscript of 't' denotes true parameter used to generate toy data

p_sparse=0.05;

%let's make these things sparse
sp_mask_t=binornd(1, p_sparse, [M, r]);

U_t=randn([M, r]);
V_t=randn([r, N]);



U_t=U_t./normr(U_t);
V_t=V_t./normc(V_t);

s_t=rand([r,1]);
s_t=sort(s_t);

S_t=zeros([r, r]);
for i=1:r
    S_t(i,i)=s_t(i);
end


P_t=U_t*S_t*V_t;

%%
%functions
grad_a=@(xs, opts) reshape(...
    (reshape(xs{1}, [opts.M, opts.r])*reshape(xs{2}, [opts.r, opts.N])-opts.Y)...
    *reshape(xs{2}, [opts.r, opts.N])',...
    [opts.M*opts.r, 1]);

grad_b=@(xs, opts) reshape(...
    reshape(xs{1}, [opts.M, opts.r])'*...
    (reshape(xs{1}, [opts.M, opts.r])*reshape(xs{2}, [opts.r, opts.N])-opts.Y),...
    [opts.r*opts.N, 1]);

grads={grad_a; grad_b};

prox_a=@(v, opts) sign(v).*max(0.0, abs(v)-opts.lam);

prox_b=@(v, opts) v./(1.0+opts.lam);

proxs={prox_a; prox_b};


lip_a = @(xs, opts) norm(xs{2}).^2;
lip_b = @(xs, opts) norm(xs{1}).^2;

lips={lip_a; lip_b};

%optimization params
nit=1e3; %max iterations before termination
tol=1e-4; %error threshold for termination

opts={};
opts.lam=0.00001;
opts.r=r;
opts.M=M;
opts.N=N;
opts.Y=P_t;

opts.nit=nit;
opts.tol=tol;
opts.lips={lip_a; lip_b};

xs = ipalm(grads, proxs, [M*r; r*N], opts);


P_h=reshape(xs{1}, [M,r])*reshape(xs{2}, [r,N]);

%%
cmi=min([P_t(:);P_h(:)]);
cma=max([P_t(:);P_h(:)]);

figure
imagesc(P_h)
set(gca,'clim',[cmi,cma])
colorbar

figure
imagesc(P_t)
set(gca,'clim',[cmi,cma])
colorbar

figure
imagesc(P_t-P_h)
set(gca,'clim',[cmi,cma])
colorbar





