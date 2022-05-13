#
# This file is a forked copy from https://github.com/bkamins/ABCDGraphGenerator.jl/
#
# The abcd_sampler.jl is licensed under the MIT License
# Copyright (c) 2019 Bogumił Kamiński, Paweł Prałat, François Théberge

using Pkg
using ABCDGraphGenerator
using Random

@info "Usage: julia abcd_sampler.jl config_filename"
@info "For the syntax of config_filename see example_config.toml file"

conf = Pkg.TOML.parsefile("config.toml")
isempty(conf["seed"]) || Random.seed!(parse(Int, conf["seed"]))

μ = haskey(conf, "mu") ? parse(Float64, conf["mu"]) : nothing
ξ = haskey(conf, "xi") ? parse(Float64, conf["xi"]) : nothing
if !(isnothing(μ) || isnothing(ξ))
    throw(ArgumentError("inconsistent data: only μ or ξ may be provided"))
end

n = parse(Int, conf["n"])

τ₁ = parse(Float64, conf["t1"])
d_min = parse(Int, conf["d_min"])
d_max = parse(Int, conf["d_max"])
d_max_iter = parse(Int, conf["d_max_iter"])
@info "Expected value of degree: $(ABCDGraphGenerator.get_ev(τ₁, d_min, d_max))"
degs = ABCDGraphGenerator.sample_degrees(τ₁, d_min, d_max, n, d_max_iter)
open(io -> foreach(d -> println(io, d), degs), conf["degreefile"], "w")

τ₂ = parse(Float64, conf["t2"])
c_min = parse(Int, conf["c_min"])
c_max = parse(Int, conf["c_max"])
c_max_iter = parse(Int, conf["c_max_iter"])
@info "Expected value of community size: $(ABCDGraphGenerator.get_ev(τ₂, c_min, c_max))"
coms = ABCDGraphGenerator.sample_communities(τ₂, c_min, c_max, n, c_max_iter)
open(io -> foreach(d -> println(io, d), coms), conf["communitysizesfile"], "w")

isCL = parse(Bool, conf["isCL"])
islocal = haskey(conf, "islocal") ? parse(Bool, conf["islocal"]) : false

p = ABCDGraphGenerator.ABCDParams(degs, coms, μ, ξ, isCL, islocal)
edges, clusters = ABCDGraphGenerator.gen_graph(p)
prefix = conf["person_id_prefix"]
open(conf["networkfile"], "w") do io
    for (a, b) in sort!(collect(edges))
        println(io, prefix, a, ",", prefix, b)
    end
end
open(conf["communityfile"], "w") do io
    for (i, c) in enumerate(clusters)
        println(io, i, ",", c)
    end
end
