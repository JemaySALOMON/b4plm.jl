
module b4plm

export  simulDBVSBV, createIndivID, mkTrial, Crossing
export@sepcies, @design, @check

# dependencies
using LinearAlgebra, 
using Random
using Printf, Distributions
using DataFrames


using Distributions, LinearAlgebra, Random, Printf, DataFrames

abstract type Species end
abstract type FieldTrial end
abstract type PlantSpecies <: Species end


############################### check packages############################################################
function checkPkg(pkg::String)
    try
        @eval using $(Symbol(pkg))
        println("Package '$pkg' is already installed.")
    catch e
        if isa(e, ArgumentError) || isa(e, LoadError)
            println("Installing package '$pkg'...")
            Pkg.add(pkg)
            @eval using $(Symbol(pkg))
        else
            rethrow(e)
        end
    end
end

macro check(pkg)
    pkgName = string(pkg)
    return quote
        import Pkg
        checkPkg($pkgName)
    end
end

############################################  PlantSpecies ##########################################

mutable struct Plant{P<:PlantSpecies}
    name::String
    id::Int
    Individuals::Int
    sigma_DBV::Float64
    sigma_SBV::Float64
    cor_DBV_SBV::Float64
    DBV_Heritability::Float64
    SBV_Heritability::Float64
    SelectedGenos::Union{Nothing, Dict{String, Dict{String, Float64}}}
    SubPop::Union{Nothing, Dict{String, Dict{String, Dict{String, Float64}}}}
    matDBVSBV::Union{Nothing, Dict{String, Any}}  # Holds "DBV_SBV", "IDs", "VarCov"
    MeanDBVSBV::Union{Nothing, Matrix{Float64}}
    iSelection::Float64
end


"""
    @species SpeciesName

Defines a new plant species type that inherits from the abstract `Plant` type and can be used to simulate Direct Breeding Values (DBV) and Social Breeding Values (SBV).

# Arguments
- `SpeciesName`: The name of the species you want to define (e.g., `Pea`, `Wheat`, etc.).

This macro dynamically generates a new subtype of `Plant` with the following fields:
- `name::String`
- `id::Int`
- `Individuals::Int`
- `sigma_DBV::Float64`
- `sigma_SBV::Float64`
- `cor_DBV_SBV::Float64`
- `DBV_Heritability::Float64`
- `SBV_Heritability::Float64`

# Example of usage

```julia
julia> @species Pea

julia> p = Pea(
           name = "Pea",
           id=1,
           Individuals=10,
           sigma_DBV=0.4,
           sigma_SBV=0.2,
           cor_DBV_SBV=0.1,
           DBV_Heritability=0.5,
           SBV_Heritability=0.3
       )
```
"""
macro species(name)
    quote
        struct $(esc(name)) <: PlantSpecies end

        function (::Type{$(esc(name))})(; name,
                                        id,
                                        Individuals,
                                        sigma_DBV,
                                        sigma_SBV,
                                        cor_DBV_SBV,
                                        DBV_Heritability,
                                        SBV_Heritability,
                                        SelectedGenos::Union{Nothing, Dict{String, Dict{String, Float64}}}=nothing,
                                        SubPop::Union{Nothing, Dict{String, Dict{String, Float64}}}=nothing,
                                        matDBVSBV::Union{Nothing, Dict{String, Any}}=nothing,
                                        MeanDBVSBV::Union{Nothing, Matrix{Float64}} = zeros(Individuals, 2),
                                        iSelection)

            Plant{$(esc(name))}(name,
                                id,
                                Individuals,
                                sigma_DBV,
                                sigma_SBV,
                                cor_DBV_SBV,
                                DBV_Heritability,
                                SBV_Heritability,
                                SelectedGenos,
                                SubPop,
                                matDBVSBV,
                                MeanDBVSBV,
                                iSelection)
        end
    end
end


# Custom display for Plant
function Base.show(io::IO, p::Plant{P}) where {P<:PlantSpecies}
    println(io, "Plant of species $(p.name):")
    println(io, "  ID: ", p.id)
    println(io, "  Individuals: ", p.Individuals)
    println(io, "  σ²_DBV: ", p.sigma_DBV, " (H²: ", p.DBV_Heritability, ")")
    println(io, "  σ²_SBV: ", p.sigma_SBV, " (H²: ", p.SBV_Heritability, ")")
    println(io, "  Corr(DBV, SBV): ", p.cor_DBV_SBV)
    println(io, "  iSelection: ", p.iSelection)
end


############################################  TRIAL ##########################################
# Abstract base type
abstract type FieldTrial end

# Field trial design type
mutable struct Design{D<:FieldTrial}
    id::Int                          # Trial ID
    nbPlots::Int                     # Number of plots
    nbBlocks::Int                    # Number of blocks
    blockEffs::Dict{String, Float64} # Block effects by block name
end

"""
    @trial DesignName

Defines a new field trial design type that inherits from the abstract `FieldTrial` type. This macro allows you to create specific trial types that can be used for experimental designs with different block effects and structures.

# Arguments
- `DesignName`: The name of the trial design you want to define (e.g., `Inter_only`, `SplitPlotTrial`, etc.).

This macro dynamically generates a new subtype of `FieldTrial` and automatically creates a constructor for the specific trial type with the following parameters:
- `id::Int`: Trial ID
- `nbPlots::Int`: Number of plots in the trial
- `nbBlocks::Int`: Number of blocks in the trial
- `blockEffs::Dict{String, Float64}`: A dictionary mapping block names to block effects (e.g., `Dict("A" => 0.5, "B" => 0.3)`)

The macro also generates a specialized `Design` struct that stores the trial design parameters.

# Example of usage

```julia
julia> @design inter_only
julia> blockEffs = Dict("A" => 0.5, "B" => 0.3, "C" => -0.1)
julia> my_design = inter_only(1, 100, 3, blockEffs)
julia> println(my_design)
```
"""
macro design(name)
    quote
        struct $(esc(name)) <: FieldTrial end

        # Constructor for the specific trial type
        function $(esc(name))(;id, nbPlots, nbBlocks, blockEffs)
            Design{$(esc(name))}(id, nbPlots, nbBlocks, blockEffs)
        end
    end
end

# Enhanced display method
function Base.show(io::IO, d::Design)
    # Print trial design information
    println(io, "Design ID: ", d.id)
    println(io, "Number of Plots: ", d.nbPlots)
    println(io, "Number of Blocks: ", d.nbBlocks)

    # Print block effects with sorted block names
    println(io, "Block Effects:")
    for b in sort(collect(keys(d.blockEffs)))
        println(io, "  Block $b => $(d.blockEffs[b])")
    end

    # Print trial type (generic or specific to the trial type)
    trial_type = typeof(d)  # Get the type of the trial (e.g., inter_only)
    println(io, "Trial Type: ", trial_type)
end

################################ FUN ######################################

# Function to create unique individual IDs
"""
    createIndivID(species)

Generate unique individual identifiers for each plant in a given `Plant` species.

# Arguments
- `species::Plant`: A `Plant` struct containing the name and number of individuals.

# Returns
- `Vector{String}`: A vector of individual IDs in the format `"SpeciesName001", "SpeciesName002", ..."`

# Example of usage

```julia
julia>  p = Pea(
        name="Pea",
        id=1,
        Individuals=3,
        sigma_DBV=0.4,
        sigma_SBV=0.2,
        cor_DBV_SBV=0.1,
        DBV_Heritability=0.5,
        SBV_Heritability=0.3)

julia> ids = createIndivID(p)
```
"""
function createIndivID(species::Plant)
    n = species.Individuals
    return [@sprintf("%s%03d", species.name, i) for i in 1:n]
end


"""
    function simulDBVSBV(species, kinship)

Simulates Direct Breeding Values (DBV) and Social Breeding Values (SBV) for a given plant species.

# Arguments
- `species`: A `Plant` instance containing genetic parameters such as variances and correlation between DBV and SBV.
- `kinship`: Optional kinship matrix. If `nothing`, an identity matrix is used by default.

# Returns
A `Dict` with the following entries:
- `"DBV_SBV"`: A matrix of simulated DBV and SBV values (n × 2), rounded to 4 decimals.
- `"IDs"`: A vector of individual IDs.
- `"VarCov"`: The 2×2 genetic variance-covariance matrix used in the simulation.

# Example 
```julia
julia> @species Pea  
julia> p = Pea(
        id=1,
        Individuals=10,
        sigma_DBV=0.4,
        sigma_SBV=0.2,
        cor_DBV_SBV=0.1,
        DBV_Heritability=0.5,
        SBV_Heritability=0.3
    )  
    result = simulDBVSBV(p)  
    result["DBV_SBV"]  
```
"""
function simulDBVSBV(species::Plant, kinship::Union{Nothing, Matrix{Float64}}=nothing)
    
    n = species.Individuals
    if kinship === nothing
        kinship = Matrix{Float64}(I, n, n)  # Identity matrix of size n x n
    end
    
    # IDs
    ids = createIndivID(species)

    # Mean matrix M (n x 2)
    M = species.MeanDBVSBV
    rownames = ids
    colnames = ["DBV", "SBV"]

    # Covariance matrix G (2 x 2)
    σ2_DBV = species.sigma_DBV
    σ2_SBV = species.sigma_SBV
    ρ = species.cor_DBV_SBV

    cov_DBV_SBV = ρ * sqrt(σ2_DBV * σ2_SBV)
    G = [σ2_DBV cov_DBV_SBV;
        cov_DBV_SBV σ2_SBV]

    # Matrix normal draw
    Z = randn(n, 2)
    L_K = cholesky(kinship).L
    L_G = cholesky(G).L
    U = L_K * Z * transpose(L_G)

    # Round to 3 decimals and attach IDs
    U_rounded = round.(U, digits=4)
    DBV_SBV = Dict("DBV_SBV" => U_rounded, "IDs" => ids, "VarCov" => G)
    return DBV_SBV
end

"""
    mkTrial(focal, tester, design, options)

Creates a trial layout for evaluating genotype combinations between a focal species and a tester species,
according to a specified design. The function returns a `DataFrame` containing all the plots of the trial.

# Arguments
- `focal`: A `Plant` object representing the focal species (e.g., wheat genotypes).
- `tester`: A `Plant` object representing the tester species (e.g., pea genotypes).
- `design`: A `Design` object that defines the trial layout, including:
  - `nbPlots`: Total number of plots in the trial.
  - `nbBlocks`: Number of blocks in the design.
  - `blockEffs`: Block labels and their effects (used to assign blocks).
- `options`:
  - `"full_factorial"`: Use a full factorial design (every focal genotype crossed with every tester).
  - `Int`: Number of times each focal genotype should be combined with randomly chosen testers (with replacement).

# Returns
A `DataFrame` with columns:
- `Focal`: Identifier of the focal genotype.
- `Tester`: Identifier of the tester genotype.
- `Block`: Assigned block label for the plot.

# Example

```julia
julia> df = mkTrial(focal=species_wheat, tester=species_pea, design=design, options=6)

```
"""
function mkTrial(; focal::Plant, tester::Plant, design::Design, options::Union{String, Int})
    
    # Step 1: Generate individual IDs for focal and tester species
    focal_ids = createIndivID(focal)
    tester_ids = createIndivID(tester)

    # Step 2: Handle full factorial or specific combinations based on options
    combinations = []

    if options == "full_factorial"
        # Create all combinations between focal and tester
        combinations = [(f, t) for f in focal_ids, t in tester_ids]
    
    elseif isa(options, Int)
        # Repeat each focal genotype `options` times with randomly selected testers
        n_testers = length(tester_ids)
    
        for f in focal_ids
            # Sample testers with replacement for each focal genotype
            chosen_testers = rand(tester_ids, options)
            for t in chosen_testers
                push!(combinations, (f, t))
            end
        end
    
    else
        throw(ArgumentError("Invalid options. Use 'full_factorial' or an integer for repetitions."))
    end


    # Step 3: Calculate expected nbPlots for the full factorial
    nbFocal = length(focal_ids)  # Number of focal individuals
    nbTester = length(tester_ids)  # Number of tester individuals
    nbBlocks = design.nbBlocks  # Number of blocks from the design

    # Full factorial design, number of plots = focal x tester x blocks
    expected_nbPlots = nbFocal * nbTester * nbBlocks

    # Step 4: Handle user-defined nbPlots for non-full factorial designs
    if options == "full_factorial"
        # Check if user-defined nbPlots matches expected number of plots
        if design.nbPlots < expected_nbPlots
            throw(ArgumentError("Mismatch: Expected nbPlots = $(expected_nbPlots), but user-defined nbPlots = $(design.nbPlots). Ensure sufficient resources for the trial."))
        end
    else
        # For non-full factorial designs, let's check based on the specific combinations and blocks
        expected_non_full_factorial_plots = length(combinations) * nbBlocks
        if design.nbPlots < expected_non_full_factorial_plots
            throw(ArgumentError("Mismatch: Expected nbPlots = $(expected_non_full_factorial_plots), but user-defined nbPlots = $(design.nbPlots). Ensure sufficient resources for the trial."))
        end
    end

    # Step 3: Organize the trial design into blocks using blockEffs from the Design class
    total_combinations = length(combinations)
    
    # Extract block names from the design's block effects
    blockEffs = design.blockEffs
    block_labels = collect(keys(blockEffs))
    
    # Ensure there are enough block labels for the total combinations
    if length(block_labels) < nbBlocks
        throw(ArgumentError("Insufficient block labels in 'blockEffs'. Ensure there are enough labels for the design"))
    end
    
    # Step 4: Ensure blocks are distributed across combinations
    blocks = repeat(block_labels, outer=div(total_combinations, length(block_labels)))
    
    # Handle remainder if the combinations don't divide evenly into blocks
    remainder = total_combinations % length(block_labels)
    if remainder != 0
        append!(blocks, block_labels[1:remainder])
    end

    # Step 5: Construct a matrix with a row for each element (Focal, Tester, Block)
    trial_matrix = []
    for (i, comb) in enumerate(combinations)
        # Using a NamedTuple to store Focal, Tester, and Block information
        push!(trial_matrix, (Focal=comb[1], Tester=comb[2], Block=blocks[i]))  # (NamedTuple)
    end

    # Step 6: Convert NamedTuples to DataFrame for structured output
    df = DataFrame(trial_matrix)  # DataFrame automatically interprets NamedTuples as columns

    # Return the DataFrame
    return df
end

##########################################################################################

function updateSpecies!(species::Plant, nbCrosses, nbProgeny)
    # Update total number of individuals
    species.Individuals = nbProgeny * nbCrosses

    # Initialize vectors to store family mean DBVs and SBVs
    mean_DBVs = Float64[]
    mean_SBVs = Float64[]

    # Loop through each subpopulation (cross)
    for (_, progeny_dict) in species.SubPop
        # Extract DBVs and SBVs from all progeny in this family
        DBVs = [ind["DBV"] for (_, ind) in progeny_dict]
        SBVs = [ind["SBV"] for (_, ind) in progeny_dict]

        # Compute and store the mean DBV and SBV for the family
        push!(mean_DBVs, round(mean(DBVs), digits=2))
        push!(mean_SBVs, round(mean(SBVs), digits=2))
    end

    # Compute inter-family variance (variance of the family means)
    species.sigma_DBV = round(var(mean_DBVs), digits=2)
    species.sigma_SBV = round(var(mean_SBVs), digits=2)

    # Optional: print the computed means for debug
   # println("Mean DBVs inter family: ", mean_DBVs)
   # println("Mean SBVs inter family: ", mean_SBVs)
end

"""
    Crossing(species, nbCrosses, nbProgeny, var_DBV=0.05, var_SBV=0.05)

Simulates a crossing scheme among selected genotypes for a given `Plant` species.

# Arguments
- `species`: A plant object containing selected genotypes in `SelectedGenos`.
- `nbCrosses`: Number of **distinct pairwise crosses** to generate from the selected genotypes.
- `nbProgeny`: Number of progeny to simulate per cross.
- `var_DBV`: Environmental or residual variance used to sample `DBV` values for the progeny.
- `var_SBV`: Environmental or residual variance used to sample `SBV` values for the progeny.

# Returns
- The same `Plant` object with its `SubPop` field updated. For each cross, progeny individuals are simulated with DBV and SBV values sampled from normal distributions centered on the mid-parent value.

# Genetic Rationale
This function is based on the concept of **"cross utility value"**, which evaluates the potential value of a cross based on the average genetic merit of the parents. The DBV (Direct Breeding Value) and SBV (Specific Breeding Value) of the progeny are drawn from normal distributions centered on the mean of the two parental values, with added variance to represent within-family variation.

Reference:
> Gallais, A. (2021). Méthodes de création de variétés en amélioration des plantes. Editions Quae.

# Example

```julia
julia> updated_species = Crossing(
        species = my_species,
        nbCrosses = 3,
        nbProgeny = 5,
        var_DBV = 0.1,
        var_SBV = 0.1
)
```
"""
function Crossing(; species::Plant, nbCrosses::Int, nbProgeny::Int)
   
    # Initialize or reset the SubPop dictionary if it's nothing
    species.SubPop = Dict{String, Dict{String, Dict{String, Float64}}}()
    
    # Loop through the selected genotypes to make crosses
    selected_genos = species.SelectedGenos
    if selected_genos == nothing || isempty(selected_genos)
        error("No selected genotypes available for crossing!")
    end
    
    # List the selected genotypes for crossing (pairing them randomly)
    geno_keys = collect(keys(selected_genos))
    n_genos = length(geno_keys)
    
    # Calculate the maximum number of possible pairs
    max_pairs = n_genos * (n_genos - 1) ÷ 2
    if nbCrosses > max_pairs
        error("Number of crosses exceeds the maximum number of possible crosses: $max_pairs")
    end
    
    # Generate all unique pairs of genotypes
    crosses = []
    for i in 1:length(geno_keys)-1
        for j in i+1:length(geno_keys)
            push!(crosses, (geno_keys[i], geno_keys[j]))
        end
    end
    
    # Randomly select the number of crosses specified by nbCrosses
    selected_crosses = rand(1:length(crosses), nbCrosses)
    
    # Perform crosses for each selected pair
    for i in selected_crosses
        parent1_name, parent2_name = crosses[i]
        
        # Get the DBV and SBV values of the selected genotypes
        parent1_data = selected_genos[parent1_name]
        parent2_data = selected_genos[parent2_name]
        
        # Create a progeny name from parent names
        progeny_name = "$(parent1_name)_$(parent2_name)"
        
        # Initialize the dictionary for this cross if it doesn't exist
        if !haskey(species.SubPop, progeny_name)
            species.SubPop[progeny_name] = Dict()
        end
        
        # Generate progeny individuals
        for k in 1:nbProgeny
            # Calculate the offspring's DBV and SBV based on the parents' DBV and SBV (mean + random noise)
            Mean_DBV = (parent1_data["DBV"] + parent2_data["DBV"]) / 2
            Mean_SBV = (parent1_data["SBV"] + parent2_data["SBV"]) / 2 
            
            progeny_DBV =  rand(Normal(Mean_DBV, species.sigma_DBV))
            progeny_SBV =  rand(Normal(Mean_SBV, species.sigma_SBV))

            # Create a dictionary for the individual progeny's DBV and SBV
            progeny_individual = Dict("DBV" => progeny_DBV, "SBV" => progeny_SBV)
            
            # Add the progeny individual to the SubPop under the respective cross
            species.SubPop[progeny_name][string("Progeny_", k)] = progeny_individual
        end
    end
    
   #update species
    updateSpecies!(species, nbCrosses, nbProgeny)
    
    
    return species  # Return the updated species with updated SubPop
end

end