function xs2=ipalm(grads, proxs, dims, opts)
    %%% inertial proximal alternating linearized minimization (iPALM)
    %%% with fixed step size and without Lipschitz computation
    %%% to solve the unconstrained optimization of form:
    %%%        min_{xi} f(x1, ..., xi, ..., xn) + sum_i h_i(xi)
    %%% using gradient functions and proximal operators
    %           
    % inputs:
    %   grads          cell of function handles, which evaluate gradient of f w.r.t. xi
    %                    use: grad_i(xs, opts)
    %                         where xs is cell of variable blocks, {x1; ...; xi; ...; xn}
    %   proxs          cell of function handles, which evaluate proximity operator of h_i
    %                    use: prox_i(v, opts)
    %                         where v has same dim as xi
    %   dims           vector of ints, sizes of vector xi
    %   opts           struct, optimization algorithm parameters as well as any other problem parameters
    %    opts.nit       int, max number of iterations to perform before termination
    %    opts.tol       float, error tolerance for termination [default: 1e-8]
    %    opts.xs0       cell, initial points for xi0 [default: random normal vector]
    %    opts.lips      cell, contains function handles to compute partial Lipschitz constants
    %                    use: lip_i(xs, opts)
    %
    % outputs:
    %   xs2            final xis estimated
    
    nblocks=length(grads);
    if isfield(opts, 'lips'); lips=opts.lips; else lips=mat2cell(ones(nblocks,1)); end;
    if isfield(opts, 'nit'); nit=opts.nit; else nit=1e3; end
    if isfield(opts, 'tol'); tol=opts.tol; else tol=1e-8; end
    
    
    %initialize vars
    if isfield(opts, 'xs0'); xs0=opts.xs0;
    else
        xs0=cell(nblocks);
        for i=1:nblocks
            xs0{i}=randn([dims(i), 1]);
        end
    end
    xs1=xs0;
    xs2=xs0;
    
    it=0;
    err=inf;
    while it<nit && err>tol
        inertia = it/(it+3.0);
        for i=1:nblocks
            % take proximal step in block i
            yi = xs1{i} + inertia * (xs1{i} - xs0{i});
            zi = xs1{i} + inertia * (xs1{i} - xs0{i});
            
            lip_i = lips{i};
            if isa(lip_i, 'function_handle')
                tau_i = lip_i(xs2, opts);
            elseif isa(lip_i, 'numeric')
                tau_i = lip_i;
            else
                error('lips must contain either numbers or function handles');
            end
                
            prox_i = proxs{i};
            grad_i = grads{i};
            
            xs2{i} = zi;
            xs2{i} = prox_i(yi - 1.0/tau_i * grad_i(xs2, opts), opts);
        end
        
        % compute relative error in x
        x1 = cell2vec(xs1, dims);
        x2 = cell2vec(xs2, dims);
        err = norm(x1-x2)/(1e-10+norm(x1));
        
        %update states for inertia
        xs0 = xs1;
        xs1 = xs2;
        
        it=it+1;
    end
    
    
    
end

function v=cell2vec(c, dims)
    n=length(c);
    cs=[0; cumsum(dims(:))];
    v=zeros([cs(end), 1]);
    for i=1:n
        v(cs(i)+1:cs(i+1))=c{i};
    end
end