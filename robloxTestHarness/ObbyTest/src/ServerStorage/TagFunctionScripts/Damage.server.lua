-- Set part color. For now, this is cosmetic, the agent doesn't see it.
local RunService = game:GetService("RunService")
local CollectionService = game:GetService("CollectionService")

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
    script.Parent.Touched:connect(function(hit)
        if hit and hit.Parent and not CollectionService:HasTag(hit.Parent, "NoDamage") and ((hit.Parent:FindFirstChild("Humanoid") or hit.Parent:FindFirstChild("NPCHumanoid"))) then
            if hit.Parent:FindFirstChild("Humanoid") then
                local pMax = script.Parent.Position + script.Parent.size * 0.5
                local pMin = script.Parent.Position - script.Parent.size * 0.5
    
                if not hit.Parent:FindFirstChild("HumanoidRootPart") then
                    return
                end

                local localPos = hit.Parent.HumanoidRootPart.Position
                local trueHit = true -- Set to false to enable more lenient hit checking below.
    
                -- Slightly more forgiving collision logic.
                local xoverMin = pMin.X < localPos.X
                local xunderMax = pMax.X > localPos.X
    
                local zoverMin = pMin.Z < localPos.Z
                local zunderMax = pMax.Z > localPos.Z
    
                trueHit = trueHit or ((zoverMin and zunderMax) and (xoverMin and xunderMax)) 
                
                if trueHit then
                    wait(0.25)
                    if hit.Parent then
                        hit.Parent.Humanoid.Health = math.max(0, hit.Parent.Humanoid.Health - script.Parent:GetAttribute("damage"))
                    end
                end
            else
                if hit.Parent then
                    hit.Parent.NPCHumanoid.Health = math.max(0, hit.Parent.Humanoid.Health - script.Parent:GetAttribute("damage"))
                end
            end
        end
    end)
end)