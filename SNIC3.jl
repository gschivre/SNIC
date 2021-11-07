using Plots, Images, TestImages, MappedArrays, StaticArrays, ProgressMeter, BenchmarkTools

# SNIC queue element
struct QueueElt_SNIC
    pos::Int        # pixel/voxel position
    lab::Int        # predicted label
    dst::Float64    # distance from the centroids associated to the label
end
Base.isless(a::QueueElt_SNIC, b::QueueElt_SNIC) = a.dst < b.dst

# SNIC queue
struct Queue_SNIC
    # PriorityQueue to get easy access to element with smallest distance to the centroid
    tree::Vector{QueueElt_SNIC}
    handle::Vector{Int} # Handle map to linear indices of pixel/voxel in the image
    buf::MVector{2, QueueElt_SNIC} # buffer to avoid allocation

    function Queue_SNIC(n::Int, v::Vector{QueueElt_SNIC})
        # at queue initialization all entry have the same priority so no need to sort them
        H = zeros(Int, n)
        for (k, e) in Iterators.enumerate(v)
            H[e.pos] = k
        end
        new(copy(v), H, MVector{2, QueueElt_SNIC}(undef))
    end
end
@inline Base.length(h::Queue_SNIC) = length(h.tree)
@inline Base.isempty(h::Queue_SNIC) = isempty(h.tree)
@inline Base.isassigned(h::Queue_SNIC, i::Int) = h.handle[i] > 0

function Base.push!(h::Queue_SNIC, v::QueueElt_SNIC)
    # check if the QueueElt_SNIC allready exist
    if !isassigned(h, v.pos) # QueueElt_SNIC doesn't exist so push it
        push!(h.tree, v)
        percUp!(h, length(h))
    else
        if v < h.tree[h.handle[v.pos]] # modify the QueueElt_SNIC only if the distance is shorter
            h.tree[h.handle[v.pos]] = v
            percUp!(h, h.handle[v.pos])
        end
    end
end

function Base.pop!(h::Queue_SNIC)
    h.buf[2] = h.tree[1]
    h.handle[h.buf[2].pos] = 0
    h.buf[1] = pop!(h.tree)
    isempty(h) || percDown!(h, 1)
    return h.buf[2]
end

function percUp!(h::Queue_SNIC, i::Int)
    h.buf[1] = h.tree[i]
    parent = i >>> 1
    @inbounds while parent > 0
        # Compare to the parent and swap if needed
        if h.buf[1] < h.tree[parent]
            h.handle[h.tree[parent].pos] = i
            h.tree[i] = h.tree[parent]
            i = parent
            parent >>>= 1
        else
            break
        end
    end
    h.handle[h.buf[1].pos] = i
    h.tree[i] = h.buf[1]
    return
end

function percDown!(h::Queue_SNIC, i::Integer)
    heapsize = length(h)
    l = 2 * i # left
    @inbounds while l <= heapsize
        r = 2 * i + 1 # right
        sc = r > heapsize || h.tree[l] < h.tree[r] ? l : r # smallest child
        # compare with the smallest child and swap if needed
        if h.tree[sc] < h.buf[1]
            h.handle[h.tree[sc].pos] = i
            h.tree[i] = h.tree[sc]
            i = sc
            l = 2 * i
        else
            break
        end
    end
    h.handle[h.buf[1].pos] = i
    h.tree[i] = h.buf[1]
    return
end

function SNIC(img::AbstractArray{T, N}, seeds::Vector{CartesianIndex{N}}, m::Number, δ::Tuple = pixelspacing(img)) where {T <: Union{AbstractRGB, AbstractGray, Number}, N}
    δv = SVector{N, Float64}([δ[d] for d in 1:N] ./ sum(δ)) # pixel spacing normalized to sum to 1
    axesImg = size(img)
    sz = prod(axesImg)
    K = length(seeds)
    s = K / sz # spatial distance normalization constante
    NNvect = @MVector zeros(Int, N >>> -1) # Vector for neighborhood tracking
    numNN = 0 # number of neighbors

    # initialize the map, centroids and the queue
    γ = zeros(N, K) # centroids
    L = zeros(Int, axesImg) # superpixel map
    isrgb = T <: AbstractRGB
    if isrgb
        img_view = MappedArrays.of_eltype(Float64, channelview(img))
        μ = zeros(3, K)
    else
        img_view = MappedArrays.of_eltype(Float64, img) # to avoid arithmetic overflows
        μ = zeros(K) # superpixel feature mean
    end
    n = zeros(Int, K)
    Q = Queue_SNIC(sz, [QueueElt_SNIC(cart2lin(sk, L), k, 0.0) for (k, sk) in Iterators.enumerate(seeds)])
    e = QueueElt_SNIC(1, 1, 0.0)
    c = CartesianIndex(1, 1)
    dst = 0.0

    @inbounds begin
        while !isempty(Q)
            e = pop!(Q)
            c = lin2cart(e.pos, L)
            L[c] = e.lab

            # update centroid
            n[e.lab] += 1
            for d in 1:N
                γ[d, e.lab] += (c[d] - γ[d, e.lab]) / n[e.lab]
            end

            # update feature mean
            if isrgb
                for chan in 1:3 # color channel
                    μ[chan, e.lab] += (img_view[chan, c] - μ[chan, e.lab]) / n[e.lab]
                end
            else
                μ[e.lab] += (img_view[c] - μ[e.lab]) / n[e.lab]
            end

            # check neighborhood
            numNN = get_neighbors!(NNvect, e.pos, axesImg)
            for NN in view(NNvect, 1:numNN)
                L[NN] == 0 || continue
                c = lin2cart(NN, L)

                # compute the distance
                dst = 0.0

                # spatial distance
                for d in 1:N
                    dst += ((c[d] - γ[d, e.lab]) * δv[d]) ^ 2
                end
                dst *= s

                # feature distance
                if isrgb # color image
                    let df = 0.0
                        for chan in 1:3 # color channel
                            df += (img_view[chan, c] - μ[chan, e.lab]) ^ 2
                        end
                        dst += df / m
                    end
                else # gray image or Array of Number
                    dst += ((img_view[c] - μ[e.lab]) ^ 2) / m
                end

                push!(Q, QueueElt_SNIC(NN, e.lab, dst))
            end
        end
    end
    return (L, μ, γ)
end

# get the pixel/voxel neighborhood (w.o diagonaly connected pixel/voxel) linear indexes
function get_neighbors(x::Int, axesA::NTuple{N, Int}) where N
    NNvect = @MVector zeros(Int, N >>> -1)
    lstidx = get_neighbors!(NNvect, x, axesA)
    return view(NNvect, 1:lstidx)
end

function get_neighbors!(NNvect::MVector{N2, Int}, x::Int, axesA::NTuple{N, Int}) where {N2, N} # N2 == 2 * N
    @inbounds begin
        widx = 1 # write index
        # first dimension neighbors (offset is ± 1)
        NNvect[widx] = x - 1
        widx += (x % axesA[1]) != 1
        NNvect[widx] = x + 1
        widx += (x % axesA[1]) != 0
        # other dimensions
        for (d, (o, k)) in Iterators.enumerate(Iterators.zip(Iterators.accumulate(*, axesA[1:(N - 1)]), Iterators.accumulate(*, axesA[2:N])))
            NNvect[widx] = x - o
            widx += (div(x - 1, o) % k) != 0
            NNvect[widx] = x + o
            widx += (div(x - 1, o) % k) != (axesA[d + 1] - 1)
        end
    end
    return widx - 1
end

# convert linear index to CartesianIndex (lin2cart doesn't check bounds !)
@inline function lin2cart(i::Int, A::AbstractArray)
    return @inbounds CartesianIndices(A)[i]
end

# convert CartesianIndex to linear index (cart2lin doesn't check bounds !)
@inline function cart2lin(i::CartesianIndex, A::AbstractArray)
    return @inbounds LinearIndices(A)[i]
end

@inline init_seeds(axesImg, K) = begin
    s = init_seeds_regular(axesImg, K)[:]
    length(s) == K || @warn string(length(s)) * " seeds produced"
    return s
end

# initialize seeds locations according to a regular grid
function init_seeds_regular(axesImg::NTuple{N, Int}, K::Int) where N
    @assert prod(axesImg) > K "Cannot produce more superpixels / voxels than img"
    
    if prod(axesImg) == K
        return CartesianIndices(img)
    else
        # number of pixels / voxels per superpixel in each dimensions
        S = fill(round(Int, (prod(axesImg) / K) ^ inv(N)), N)
        if any(axesImg .< S)
            sortAxesImg = sortperm(axesImg)
            for (i, d) in Iterators.enumerate(sortAxesImg)
                S[d] = axesImg[d]
                rd = sortAxesImg[(i + 1):end]
                S[rd] .= round(Int, (prod(axesImg[rd]) / K) ^ inv(N - i))
                if all(axesImg .>= S)
                    break
                end
            end
        end
        c = ceil.(Int, S ./ 2)

        return CartesianIndices(Tuple([c[d]:S[d]:axesImg[d] for d in 1:N]))
    end
end

# rectangular box from upper-left to down-right corner
function rectBox(ul::CartesianIndex{N}, dr::CartesianIndex{N}) where N
    return CartesianIndices(ntuple(d -> ul[d]:dr[d], N))
end

# K superpixel/voxel inside a rectangular box
function BoxSNIC(img::AbstractArray{T, N}, box::CartesianIndices{N}, K::Int, m::Number, δ::Tuple = pixelspacing(img)) where {T <: Union{AbstractRGB, AbstractGray, Number}, N}
    seeds = init_seeds(size(box), K)
    for i in 1:length(seeds)
        seeds[i] += box[1]
    end
    return SNIC(img, seeds, m, δ)
end

# construct the mean image from the superpixel map and the mean value
function mean_image(Map::AbstractArray{Int64, N}, μ::VecOrMat{Float64}) where N
    col = size(μ, 1)
    if col == 3
        return [RGB(μ[:, Map[x]]...) for x in CartesianIndices(Map)]
    else
        return [Gray(μ[Map[x]]) for x in CartesianIndices(Map)]
    end
end

# given a superpixel map, find its contour
function find_contour(Map::AbstractArray{Int})
    g = magnitude(imgradients(Gray.(Map ./ maximum(Map[:])), KernelFactors.ando3, "reflect")...)
    return findall(g .> 0.0)
end

# method for the gradient magnitude in 3D
function Images.magnitude(gx::AbstractArray, gy::AbstractArray, gz::AbstractArray)
    return @. sqrt(gx ^ 2 + gy ^ 2 + gz ^ 2)
end

# draw contour on an image
function draw_contour(img::AbstractArray{T}, Map::AbstractArray{Int}, Ccolor::RGB = RGB(1, 0, 0)) where T <: Union{AbstractRGB, AbstractGray, Number}
    contour = find_contour(Map)
    if (T <: AbstractGray) || (T <: Number)
        img_tmp = RGB.(img)
    elseif T <: AbstractRGB
        img_tmp = deepcopy(img)
    end
    for ci in contour
        img_tmp[ci] = Ccolor
    end
    return img_tmp
end

function stippling(img::AbstractArray{T, N}, n::Int, iter::Int = 50) where {T <: Union{AbstractGray, Number}, N}
    img_view = MappedArrays.of_eltype(Float64, img)
    ρimg = x::CartesianIndex{N} -> 1.0 - img_view[x]
    return cvt(n, size(img), ρimg, iter)[[2; 1]]
end

# build the voronoi map (using the SNIC algorithm)
function voronoi_map(g::Vector{CartesianIndex{N}}, Ω::NTuple{N, Int}, ρ = nothing, δ::Vector = ones(N)) where N
    δv = SVector{N, Float64}([δ[d] for d in 1:N] ./ sum(δ)) # domain spacing normalized to sum to 1
    K = length(g)
    sz = prod(Ω)
    L = zeros(Int, Ω)
    γ = zeros(N, K)
    n = zeros(K)
    Q = Queue_SNIC(sz, [QueueElt_SNIC(cart2lin(sk, L), k, 0.0) for (k, sk) in Iterators.enumerate(g)])
    if ρ !== nothing
        W = maximum([ρ(x) for x in CartesianIndices(Ω)])
        ρnorm =  x::CartesianIndex{N} -> ρ(x) / W
    else
        ρnorm =  x::CartesianIndex{N} -> 1.0 / prod(Ω)
    end
    voronoi_map!(L, γ, n, Q, ρnorm, δv, Ω)
    return (L, γ)
end

function voronoi_map!(L::AbstractArray{Int, N}, γ::Matrix{Float64}, n::Vector{Float64}, Q::Queue_SNIC, ρ::Any, δv::SVector{N, Float64}, Ω::NTuple{N, Int}) where N
    NNvect = @MVector zeros(Int, N >>> -1) # Vector for neighborhood tracking
    numNN = 0 # number of neighbors
    e = QueueElt_SNIC(1, 1, 0.0)
    c = CartesianIndex(1, 1)
    w = 0.0
    dst = 0.0

    # main loop
    @inbounds while !isempty(Q)
        e = pop!(Q)
        c = lin2cart(e.pos, L)
        L[c] = e.lab

        # update centroid
        w = ρ(c)
        n[e.lab] += w

        # update centroid
        for d in 1:N
            γ[d, e.lab] += w * (c[d] - γ[d, e.lab]) / n[e.lab] # weighted mean
        end

        # check neighborhood
        numNN = get_neighbors!(NNvect, e.pos, Ω)
        for NN in view(NNvect, 1:numNN)
            L[NN] == 0 || continue
            c = lin2cart(NN, L)

            # compute the distance
            dst = 0.0

            # spatial distance
            for d in 1:N
                dst += ((c[d] - γ[d, e.lab]) * δv[d]) ^ 2
            end

            push!(Q, QueueElt_SNIC(NN, e.lab, dst))
        end
    end
end

# build a centroidal voronoi tesselation
function cvt(K::Int, Ω::NTuple{N, Int}, ρ = nothing, iter::Int = 50, δ::Vector = ones(N)) where N
    if ρ !== nothing
        W = maximum([ρ(x) for x in CartesianIndices(Ω)])
        ρnorm =  x::CartesianIndex{N} -> ρ(x) / W
    else
        ρnorm =  x::CartesianIndex{N} -> 1.0 / prod(Ω)
    end
    g = init_seed_random(Ω, K, ρnorm, 1.0)
    δv = SVector{N, Float64}([δ[d] for d in 1:N] ./ sum(δ)) # domain spacing normalized to sum to 1
    K = length(g)
    sz = prod(Ω)
    L = zeros(Int, Ω)
    γ = zeros(N, K)
    n = zeros(K)
    Q = Queue_SNIC(sz, [QueueElt_SNIC(cart2lin(sk, L), k, 0.0) for (k, sk) in Iterators.enumerate(g)])

    # first iteration
    voronoi_map!(L, γ, n, Q, ρnorm, δv, Ω)
    centroids2seeds!(g, γ)
    Q = Queue_SNIC(sz, [QueueElt_SNIC(cart2lin(sk, L), k, 0.0) for (k, sk) in Iterators.enumerate(g)])

    # remaining Lloyd’s iterations
    @showprogress for _ in 2:iter
        L .= 0
        n .= 0.0
        voronoi_map!(L, γ, n, Q, ρnorm, δv, Ω)
        centroids2seeds!(g, γ)
        Q = Queue_SNIC(sz, [QueueElt_SNIC(cart2lin(sk, L), k, 0.0) for (k, sk) in Iterators.enumerate(g)])
    end
    return (L, γ, g)
end

# closest seeds to given centroids matrix
centroids2seeds(c::Matrix{Float64}) = [CartesianIndex(max.(round.(Int, ci), 1)...) for ci in eachcol(c)]
centroids2seeds!(s::Vector{CartesianIndex{N}}, c::Matrix{Float64}) where N = begin
    for (i, ci) in Iterators.enumerate(eachcol(c))
        s[i] = CartesianIndex(max.(round.(Int, ci), 1)...)
    end
end

# initialize seeds randomly
function init_seed_random(Ω::NTuple{N, Int}, K::Int, ρ = nothing, M::Float64 = ρ === nothing ? 1.0 : maximum([ρ(x) for x in CartesianIndices(Ω)])) where N
    s = Vector{CartesianIndex{N}}(undef, K)
    if ρ === nothing # uniform density
        for k in 1:K
            s[k] = CartesianIndex(ntuple(d -> rand(1:Ω[d]), N))
        end
    else
        for k in 1:K
            s[k] = r_sample(Ω, ρ, M)
        end
    end
    return s
end

# rejection sampling
function r_sample(Ω::NTuple{N, Int}, ρ, M::Float64) where N
    u = CartesianIndex(ntuple(d -> rand(1:Ω[d]), N))
    z = rand()
    if z < ρ(u) / M
        return u
    else
        return r_sample(Ω, ρ, M)
    end
end

function draw_stippling(stips)
    return scatter(stips[1][2, :], stips[1][1, :];
                size = size(stips[2]),
                xlims = (1, size(stips[2], 2)),
                ylims = (1, size(stips[2], 1)),
                legend = false,
                markersize = 1,
                yflip = true,
                axis = nothing,
                showaxis = false)
end