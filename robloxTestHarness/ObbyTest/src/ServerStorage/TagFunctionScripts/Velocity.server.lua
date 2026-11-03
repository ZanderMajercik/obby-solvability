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
    local velocityAttribute = HttpService:JSONDecode(script.Parent:GetAttribute("velocity"))
    local velocity = Vector3.new(velocityAttribute[1], velocityAttribute[2], velocityAttribute[3])
    script.Parent.AssemblyLinearVelocity = velocity
end)