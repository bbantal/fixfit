# Simulate the BOLD signal
function boldsignal(t, y)
    """
    Simulated BOLD response to input

    """

    κ = 0.65;
    α = 0.32;
    τ = 0.98;
    ρ = 0.34;
    V_0 = 0.02;
    γ = 0.41;

    bold = zeros(size(y));
    
    for roi in 1:length(y[:, 1])
        s = ones(1, length(y[1, :])).*0.1;
        f = ones(size(s)).*0.1;
        v = ones(size(s)).*0.1;
        q = ones(size(s)).*0.1;

        for i in 2:length(y[1, :])
            z = y[roi, i];
            dt = t[i]-t[i-1];
            ds = z - (κ*s[i-1])-(γ*(f[i-1]-1));
            df = s[i-1];
            dv = (f[i-1]-(v[i-1]^(1/α)))/τ;
            dq = (((f[i-1]*(1-(1-(ρ^(1/f[i-1])))))/ρ)-(((v[i-1]^(1/α))*(q[i-1]))/v[i-1]))/τ;
            s[i] = s[i-1] + ds*dt;
            f[i] = f[i-1] + df*dt;
            v[i] = v[i-1] + dv*dt;
            q[i] = q[i-1] + dq*dt;
        end
        bold[roi, :] = V_0 .* ((7 .* ρ .* (1 .- q ./ v)) .+ ((2 .* ρ .- 0.2) .* (1 .-v )));
    end

    return (bold[:, 1:800:end])
end