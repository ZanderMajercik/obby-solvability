local HttpService = game:GetService("HttpService")
local RunService = game:GetService("RunService")

DELTA_TIME = 0
local conditionsMetBindable = Instance.new("BindableEvent")
RunService.Stepped:connect(function(_, deltaTime)
    DELTA_TIME = deltaTime
    local RUN_FLAG = true
    for name, value in pairs(script:GetAttributes()) do
        attributeValue = script.Parent.Parent.States[name].Value
        -- print(name, attributeValue)
        if attributeValue == false then
            RUN_FLAG = false
            break
        end
    end
    if RUN_FLAG then
        conditionsMetBindable:Fire()
    end
end)
local conditionsMet = conditionsMetBindable.Event

conditionsMet:Connect(function()
    if not script.Parent.CustomPhysicalProperties then
        script.Parent.CustomPhysicalProperties = PhysicalProperties.new(script.Parent.Material) -- uses the material's physical properties as a basis for the part
    end
    local orig = script.Parent.CustomPhysicalProperties
    
    friction = script.Parent:GetAttribute("friction")
    frictionWeight = script.Parent:GetAttribute("frictionWeight")
    script.Parent.CustomPhysicalProperties = PhysicalProperties.new(orig.Density, friction, orig.Elasticity, frictionWeight, orig.ElasticityWeight)
end)