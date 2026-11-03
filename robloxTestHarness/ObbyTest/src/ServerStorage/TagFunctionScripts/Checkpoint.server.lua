local CollectionService = game:GetService("CollectionService")

--Report a succcessful trajectory and trigger character death.
script.Parent.Touched:connect(function(hit)
    -- Touched is an event that is fired when the object is touched
    local hum = hit.Parent:FindFirstChild("Humanoid")
    -- If a part in a model touches the part, touch = that part not the model
    if hum and hum.Health > 0 then
        -- The thing that touched the checkpoint is alive
        local character = hit.Parent
        if character then
            CollectionService:AddTag(script.Parent, character.Name)
        end
    end
end)