# test_arc.jl
# Simple code to test the Julia interface to ARC

using GALAHAD
using Test
using Printf
using Accessors
using CUTEst
using NLPModels
using LinearAlgebra

# Custom userdata struct
struct userdata_arc
  p::Float64
end

f = "BDQRTIC"
pc = "N=100"
nlp = CUTEstModel(f, "-param", pc)
name = "$(nlp.meta.name)-$(nlp.meta.nvar)"
# function test_arc(nlp)

x0 = nlp.meta.x0


# Objective function
function fun(n::Int, x::Vector{Float64}, f::Ref{Float64}, userdata::userdata_arc)
  p = userdata.p
  f[] = NLPModels.obj(nlp, x)
  return 0
end

# Gradient of the objective
function grad(n::Int, x::Vector{Float64}, g::Vector{Float64}, userdata::userdata_arc)
  p = userdata.p
  g .= NLPModels.grad(nlp, x)
  return 0
end

# Hessian-vector product
function hessprod(n::Int, x::Vector{Float64}, u::Vector{Float64}, v::Vector{Float64},
  got_h::Bool, userdata::userdata_arc)
  # save to vector u
  NLPModels.hprod!(nlp, x, v, u)
  return 0
end


# Derived types
data = Ref{Ptr{Cvoid}}()
control = Ref{arc_control_type{Float64}}()
arc_read_specfile(control, "test_galahad/ARC.template")
inform = Ref{arc_inform_type{Float64}}()

# Set user data
userdata = userdata_arc(4.0)

# Set problem data
n = nlp.meta.nvar # dimension
ne = nlp.meta.nnzh # Hesssian elements

# Set storage
g = zeros(Float64, n) # gradient
st = ' '
status = Ref{Cint}()

@printf(" Fortran sparse matrix indexing\n\n")
@printf(" tests reverse-communication options\n\n")

# reverse-communication input/output
eval_status = Ref{Cint}()
f = Ref{Float64}(0.0)
u = zeros(Float64, n)
v = zeros(Float64, n)


# Initialize ARC
arc_initialize(data, control, status)

# Set user-defined control options
@reset control[].f_indexing = true # Fortran sparse matrix indexing
# @reset control[].print_level = Cint(1)

# Start from 1.5
x = copy(nlp.meta.x0)


# access by products

st = 'P'
arc_import(control, data, status, n, "absent",
  ne, C_NULL, C_NULL, C_NULL)


while true # reverse-communication loop
  arc_solve_reverse_without_mat(data, status, eval_status,
    n, x, f[], g, u, v)
  println(status[])
  if status[] <= 0
    break
  elseif status[] == 2 # evaluate f
    eval_status[] = fun(n, x, f, userdata)
  elseif status[] == 3 # evaluate g
    eval_status[] = grad(n, x, g, userdata)
  elseif status[] == 5 # evaluate H
    eval_status[] = hessprod(n, x, u, v, false, userdata)
  elseif status[] == 6 # evaluate the product with P
    @printf(" the product with P is not defined\n")
    eval_status[] = 0
  else
    @printf(" the value %1i of status should not occur\n", status)
  end

end


arc_information(data, inform, status)

if inform[].status[] == 0
  @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st,
    inform[].iter, inform[].obj, inform[].status)
else
  @printf("%c: ARC_solve exit status = %1i\n", st, inform[].status)
end


# Delete internal workspace
arc_terminate(data, control, inform)
# end


