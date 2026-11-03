local RunService = game:GetService("RunService")

local stateFolder = script.Parent.States

touchConnection = script.Parent.Touched:connect(function(hit)
    for idx, attribute in stateFolder:GetChildren() do
        script.Parent.Material = Enum.Material.Neon
        local globalState = script.Parent.Parent.States[attribute.Name]
        globalState.Value = true --not globalState.Value
        touchConnection:Disconnect()
    end
end)

-- script.Parent.TouchEnded:connect(function(hit)
--     for idx, attribute in stateFolder:GetChildren() do
--         script.Parent.Material = Enum.Material.SmoothPlastic
--         local globalState = script.Parent.Parent.States[attribute.Name]
--         globalState.Value = false --not globalState.Value
--     end
-- end)