local RunService = game:GetService("RunService")

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

local TICK = 0.05
local db = Instance.new("BoolValue", script.Parent)
db.Name = "db"
db.Value = false

disappearDelay = script.Parent:GetAttribute("disappearDelay")
reappearDelay = script.Parent:GetAttribute("reappearDelay")

conditionsMet:Connect(function()
    script.Parent.Touched:connect(function(hit)
        local db = script.Parent:WaitForChild("db")
        if db.Value == false then
            db.Value = true
            for i = 1, disappearDelay do
                script.Parent.Transparency = i / disappearDelay
                wait(TICK)
            end
            script.Parent.CanCollide = false
            wait(reappearDelay * TICK)
            script.Parent.CanCollide = true
            script.Parent.Transparency = 0
            db.Value = false
        end
    end)
end)