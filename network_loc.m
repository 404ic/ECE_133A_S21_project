function [pos_free, cost, lambdas] = network_loc(N, E, pos_anchor, rho)
    K = 9;
    x = rand([2*(N - K) 1]);
    pos = [zeros(N-K, 2); pos_anchor];
    lambda = 0.1;
    b1 = 0.8;
    b2 = 2;
    cost = [norm(calcError(rho, E, pos, x, N, K))^2];
    lambdas = [lambda];
    while true
        error = calcError(rho, E, pos, x, N, K);
        A = calcJacobian(E, pos, N, K, x);
        % x_new = x - inv(A'*A + lambda*eye(size(A,2))) * A' * error;
        tmp = (A'*A + lambda*eye(size(A,2))) \ (A' * error);
        x_new = x - tmp;
        if norm(calcError(rho, E, pos, x_new, N, K)) < norm(calcError(rho, E, pos, x, N, K))
            x = x_new;
            lambda = b1 * lambda;
        else
            lambda = b2 * lambda;
        end
        if 2*A'*error < 1e-5
            break
        end
        cost = [cost norm(calcError(rho, E, pos, x, N, K))^2];
        lambdas = [lambdas lambda];
    end
    pos_free = [x(1:length(x)/2) x(length(x)/2+1:end)];
end

function A = calcJacobian(E,pos,N,K,x)
    pos(1:N-K,1) = x(1:N-K);
    pos(1:N-K,2) = x(N-K+1:2*(N-K));
    A = zeros(size(E,1), 2*(N-K));
    for i = 1:length(E)
        first = pos(E(i,1),:);
        second = pos(E(i,2),:);
        if E(i,1) > N - K
            d = norm(first - second);
            A(i, E(i,2)) = - (first(1) - second(1)) / d;
            A(i, E(i,2) + N - K) = - (first(2) - second(2)) / d;
        elseif E(i,2) > N - K
            d = norm(first - second);
            A(i, E(i,1)) =  (first(1) - second(1)) / d;
            A(i, E(i,1) + N - K) = (first(2) - second(2)) / d;
        else  
            d = norm(first - second);
            A(i, E(i,1)) =  (first(1) - second(1)) / d;
            A(i, E(i,2)) = - (first(1) - second(1)) / d;
            A(i, E(i,1) + N - K) = (first(2) - second(2)) / d;
            A(i, E(i,2) + N - K) = - (first(2) - second(2)) / d;
        end
    end
end

function error = calcError(rho, E, pos, x, N, K)
    pos(1:N-K,1) = x(1:N-K);
    pos(1:N-K,2) = x(N-K+1:2*(N-K));
    error = zeros(size(rho));
    for i = 1:length(E)
        error(i) = norm(pos(E(i,1),:) - pos(E(i,2),:)) - rho(i);
    end
end

